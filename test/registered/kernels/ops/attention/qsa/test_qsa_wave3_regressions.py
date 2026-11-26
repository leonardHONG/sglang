"""Regression tests covering Wave 3 QSA, PLE, Mamba state scatter, mRoPE,
Qwen2MoE dual-stream, TokenizerManager parallelism, and MTP split-prefill
DRAFT_EXTEND_V2 graph replay."""

from __future__ import annotations

import gc
import math
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


@pytest.fixture(autouse=True)
def runtime_context():
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield
    finally:
        reset_context()


class _FakeKVLayer:
    def __init__(self, k_buf: torch.Tensor, v_buf: torch.Tensor):
        self._k = k_buf
        self._v = v_buf

    def key(self):
        return self._k

    def value(self):
        return self._v

    def store(self, layer, out_cache_loc, k, v):
        del layer
        locs = out_cache_loc.long()
        self._k[locs] = k.to(self._k.dtype)
        self._v[locs] = v.to(self._v.dtype)


class _FakeLayerRouter:
    def __init__(self, kv: _FakeKVLayer):
        self._kv = kv

    def read(self, layer_id: int):
        del layer_id
        return self._kv

    def layer(self, layer_id: int):
        del layer_id
        return self._kv


def _make_qsa_backend(
    *,
    num_slots: int = 4096,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
    max_reqs: int = 64,
    max_ctx: int = 512,
    compress_ratio: int = 4,
    token_topk: int = 64,
    cuda_graph_max_tokens: int = 32,
    device: str = "cuda",
):
    torch.manual_seed(42)
    k_buf = torch.randn((num_slots, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device).to(kv_dtype)
    v_buf = torch.randn((num_slots, num_kv_heads, head_dim), dtype=torch.bfloat16, device=device).to(kv_dtype)
    kv = _FakeKVLayer(k_buf, v_buf)
    pool = SimpleNamespace(
        layer_router=_FakeLayerRouter(kv),
        qsa_compress_ratio=compress_ratio,
        qsa_compressed_page_size=16,
        qsa_token_topk=token_topk,
        qsa_block_topk=token_topk // compress_ratio,
        qsa_compressed_dtype=kv_dtype,
    )
    req_to_token = torch.zeros((max_reqs, max_ctx), dtype=torch.int32, device=device)
    for r in range(1, max_reqs):
        base = ((r * 37) % (num_slots - max_ctx - 1)) + 1
        req_to_token[r] = torch.arange(base, base + max_ctx, dtype=torch.int32, device=device)

    backend = object.__new__(QwenSparseAttnBackend)
    backend.device = torch.device(device)
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend.req_to_token = req_to_token
    backend._kv_shard_pool = None
    backend._fused_kv_pool_eligible = False
    backend._fp8_decode_workspace = None
    backend._flydsl_decode_workspace = None
    backend._trtllm_workspace = None
    backend._trtllm_counters = {}
    backend._trtllm_sparse_tables = {}
    backend._fa2_scratch = {}
    backend._mtp_shared_sparse_indices = None
    backend._cuda_graph_max_tokens = cuda_graph_max_tokens
    backend.compress_ratio = compress_ratio
    backend.max_context_len = max_ctx
    backend.qsa_profile = SimpleNamespace(
        budget=token_topk,
        block_topk=token_topk // compress_ratio,
        compress_ratio=compress_ratio,
    )
    backend.forward_metadata = None
    backend.runner = None

    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        tp_v_head_num=num_kv_heads,
        head_dim=head_dim,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=1.0 / math.sqrt(head_dim),
    )
    return backend, layer, k_buf, v_buf, req_to_token


def test_fp8_decode_workspace_never_reallocated_after_graph_capture():
    """Regression: oversized eager decode/verify (> _cuda_graph_max_tokens)
    must use a temporary per-call workspace and NEVER replace or free
    _fp8_decode_workspace held by captured CUDA graphs."""
    # M=20 with Hkv=2 => base=40 (splits=1), M=100 => base=200 (splits=4).
    # Use cuda_graph_max_tokens=72 (base=144 -> splits=4) so graph capture uses
    # _fp8_decode_workspace, then run eager M=96 (> 72, also splits=4).
    backend, layer, k_buf, v_buf, req_to_token = _make_qsa_backend(
        num_slots=8192,
        max_reqs=128,
        max_ctx=256,
        cuda_graph_max_tokens=72,
    )
    device = "cuda"
    batch_graph = 72
    batch_eager = 96
    topk_cols = 64

    q_g = torch.randn((batch_graph, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
    gate_g = torch.randn_like(q_g)
    req_g = torch.arange(1, batch_graph + 1, dtype=torch.int32, device=device)
    seq_g = torch.full((batch_graph,), 128, dtype=torch.int32, device=device)
    idx_g = torch.arange(topk_cols, dtype=torch.int32, device=device).unsqueeze(0).expand(batch_graph, -1).contiguous()
    meta_g = QwenSparseAttnMetadata(
        sequence_lengths=seq_g,
        token_to_batch_idx=torch.arange(batch_graph, dtype=torch.int32, device=device),
        token_slot_table=torch.zeros((batch_graph, 1), dtype=torch.int32, device=device),
        indexer_metadata=None,
        row_req_pool_indices=req_g,
        is_cuda_graph=True,
    )
    fb_g = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=req_g,
    )
    backend.forward_metadata = meta_g

    # Warmup allocates _fp8_decode_workspace once (before graph capture).
    ref_out = backend._forward_paged_attention(q_g, layer, fb_g, idx_g, output_gate=gate_g).clone()
    ws = backend._fp8_decode_workspace
    assert ws is not None
    assert ws[0].shape[0] == batch_graph
    assert ws[0].shape[2] == 4
    ptr0, ptr1 = ws[0].data_ptr(), ws[1].data_ptr()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        backend._forward_paged_attention(q_g, layer, fb_g, idx_g, output_gate=gate_g)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = backend._forward_paged_attention(q_g, layer, fb_g, idx_g, output_gate=gate_g)
    torch.cuda.synchronize()

    # Now run an oversized eager batch (batch_eager=96 > capacity=72, splits=4).
    q_e = torch.randn((batch_eager, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
    gate_e = torch.randn_like(q_e)
    req_e = torch.arange(1, batch_eager + 1, dtype=torch.int32, device=device)
    seq_e = torch.full((batch_eager,), 128, dtype=torch.int32, device=device)
    idx_e = torch.arange(topk_cols, dtype=torch.int32, device=device).unsqueeze(0).expand(batch_eager, -1).contiguous()
    backend.forward_metadata = QwenSparseAttnMetadata(
        sequence_lengths=seq_e,
        token_to_batch_idx=torch.arange(batch_eager, dtype=torch.int32, device=device),
        token_slot_table=torch.zeros((batch_eager, 1), dtype=torch.int32, device=device),
        indexer_metadata=None,
        row_req_pool_indices=req_e,
        is_cuda_graph=False,
    )
    fb_e = SimpleNamespace(forward_mode=ForwardMode.DECODE, req_pool_indices=req_e)
    e_out = backend._forward_paged_attention(q_e, layer, fb_e, idx_e, output_gate=gate_e)
    assert e_out.shape == (batch_eager, layer.tp_q_head_num * layer.head_dim)
    assert torch.isfinite(e_out).all()

    # Graph workspace pointers must NEVER have changed or been freed.
    assert backend._fp8_decode_workspace[0].data_ptr() == ptr0
    assert backend._fp8_decode_workspace[1].data_ptr() == ptr1

    # Poison caching-allocator free blocks and replay captured graph.
    gc.collect()
    poison = [torch.full((72, 8, 4, 128), float("nan"), dtype=torch.float32, device=device) for _ in range(4)]
    del poison
    g_out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(g_out, ref_out, atol=1e-3, rtol=1e-3)


def test_qsa_output_gate_all_paths():
    """Verify every QSA backend execution path applies output_gate identically
    to unfused_output * sigmoid(output_gate): direct FP8 (splits=1 & splits=4,
    EXPAND=True/False), fused-KV, TRT-LLM/FA2 fallback, prefill no-prefix,
    prefill with prefix, CP forward_extend, and CPU fallback."""
    device = "cuda"
    backend, layer, k_buf, v_buf, req_to_token = _make_qsa_backend(
        num_slots=8192,
        max_reqs=128,
        max_ctx=256,
        compress_ratio=4,
        token_topk=64,
        cuda_graph_max_tokens=100,
    )

    # 1) Direct FP8 decode: splits=1 (batch=4) & splits=4 (batch=72), EXPAND=False & EXPAND=True.
    for batch, use_block in ((4, False), (72, False), (4, True), (72, True)):
        q = torch.randn((batch, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
        gate_3d = torch.randn((batch, layer.tp_q_head_num, layer.head_dim * 2), dtype=torch.bfloat16, device=device)[:, :, : layer.head_dim]
        req_idx = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        seq_lens = torch.full((batch,), 128, dtype=torch.int32, device=device)
        if use_block:
            idx = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
            idx_meta = SimpleNamespace(
                decode_logical_positions=seq_lens - 1,
                get_seqlens_int32=lambda s=seq_lens: s,
            )
        else:
            idx = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
            idx_meta = None
        backend.forward_metadata = QwenSparseAttnMetadata(
            sequence_lengths=seq_lens,
            token_to_batch_idx=torch.arange(batch, dtype=torch.int32, device=device),
            token_slot_table=torch.zeros((batch, 1), dtype=torch.int32, device=device),
            indexer_metadata=idx_meta,
            row_req_pool_indices=req_idx,
            is_cuda_graph=False,
        )
        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, req_pool_indices=req_idx)
        unfused = backend._forward_paged_attention(q, layer, fb, idx, output_gate=None)
        fused = backend._forward_paged_attention(q, layer, fb, idx, output_gate=gate_3d)
        expected = (unfused.float() * torch.sigmoid(gate_3d.reshape(batch, -1).float())).to(torch.bfloat16)
        torch.testing.assert_close(fused, expected, atol=2e-2, rtol=2e-2)

    # 2) Non-direct fallback (TRT-LLM or FA2) when direct FP8 is disabled.
    batch = 4
    q = torch.randn((batch, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
    gate = torch.randn((batch, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
    req_idx = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), 64, dtype=torch.int32, device=device)
    idx = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    backend.forward_metadata = QwenSparseAttnMetadata(
        sequence_lengths=seq_lens,
        token_to_batch_idx=torch.arange(batch, dtype=torch.int32, device=device),
        token_slot_table=torch.zeros((batch, 1), dtype=torch.int32, device=device),
        indexer_metadata=None,
        row_req_pool_indices=req_idx,
        is_cuda_graph=False,
    )
    fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, req_pool_indices=req_idx)
    with mock.patch(
        "sglang.srt.layers.attention.qwen_sparse_attn_backend.can_use_sparse_gqa_paged_decode_fp8",
        return_value=False,
    ):
        unfused_fb = backend._forward_paged_attention(q, layer, fb, idx, output_gate=None)
        gated_fb = backend._forward_paged_attention(q, layer, fb, idx, output_gate=gate)
        expected_fb = (unfused_fb.float() * torch.sigmoid(gate.reshape(batch, -1).float())).to(torch.bfloat16)
        torch.testing.assert_close(gated_fb, expected_fb, atol=2e-2, rtol=2e-2)

    # 3) Fused-KV fast path in forward_decode & forward_extend.
    raw_fused = torch.randn((batch, layer.tp_q_head_num * layer.head_dim), dtype=torch.bfloat16, device=device)
    k_dec = torch.randn((batch, layer.tp_k_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
    v_dec = torch.randn((batch, layer.tp_v_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
    fb_fused = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=req_idx,
        out_cache_loc=torch.arange(1, batch + 1, dtype=torch.int64, device=device),
        attn_cp_metadata=None,
    )
    with (
        mock.patch.object(backend, "_prefers_direct_fp8_decode", return_value=False),
        mock.patch.object(
            backend, "_try_fused_kv_attention", side_effect=lambda *a, **kw: raw_fused.clone()
        ),
    ):
        got_dec = backend.forward_decode(
            q, k_dec, v_dec, layer, fb_fused, save_kv_cache=True, topk_indices=idx, output_gate=gate
        )
        got_ext = backend.forward_extend(
            q, k_dec, v_dec, layer, fb_fused, save_kv_cache=True, topk_indices=idx, output_gate=gate
        )
        expected_f = (raw_fused.float() * torch.sigmoid(gate.reshape(batch, -1).float())).to(torch.bfloat16)
        torch.testing.assert_close(got_dec, expected_f, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(got_ext, expected_f, atol=2e-2, rtol=2e-2)

    # 4) CP path in forward_extend.
    fb_cp = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=SimpleNamespace(),
    )
    with (
        mock.patch("sglang.srt.layers.attention.qwen_sparse_attn_backend.is_cp_active", return_value=True),
        mock.patch.object(backend, "_forward_extend_cp", return_value=raw_fused.clone()),
    ):
        got_cp = backend.forward_extend(
            q, k_dec, v_dec, layer, fb_cp, save_kv_cache=False, topk_indices=idx, output_gate=gate
        )
        torch.testing.assert_close(got_cp, expected_f, atol=2e-2, rtol=2e-2)

    # 5) Prefill extend (no-prefix & with-prefix) on CUDA.
    for prefix_len, extend_len in ((0, 32), (16, 16)):
        total_len = prefix_len + extend_len
        q_p = torch.randn((extend_len, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
        k_p = torch.randn((extend_len, layer.tp_k_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
        v_p = torch.randn((extend_len, layer.tp_v_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
        gate_p = torch.randn_like(q_p)
        out_locs = req_to_token[1, prefix_len:total_len].long()
        topk_p = torch.full((extend_len, 64), -1, dtype=torch.int32, device=device)
        for r in range(extend_len):
            pos_end = prefix_len + r + 1
            topk_p[r, :pos_end] = torch.arange(pos_end, dtype=torch.int32, device=device)
        backend.forward_metadata = QwenSparseAttnMetadata(
            sequence_lengths=torch.tensor([total_len], dtype=torch.int32, device=device),
            token_to_batch_idx=torch.zeros(extend_len, dtype=torch.int32, device=device),
            token_slot_table=req_to_token[1:2, :total_len].contiguous(),
            indexer_metadata=None,
            row_req_pool_indices=torch.ones(extend_len, dtype=torch.int32, device=device),
            is_cuda_graph=False,
        )
        fb_p = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=None,
            out_cache_loc=out_locs,
            extend_seq_lens=torch.tensor([extend_len], dtype=torch.int32, device=device),
            extend_seq_lens_cpu=[extend_len],
            seq_lens=torch.tensor([total_len], dtype=torch.int32, device=device),
            seq_lens_cpu=[total_len],
            req_pool_indices=torch.tensor([1], dtype=torch.int32, device=device),
        )
        unfused_p = backend.forward_extend(
            q_p, k_p, v_p, layer, fb_p, save_kv_cache=True, topk_indices=topk_p, output_gate=None
        )
        gated_p = backend.forward_extend(
            q_p, k_p, v_p, layer, fb_p, save_kv_cache=True, topk_indices=topk_p, output_gate=gate_p
        )
        expected_p = (unfused_p.float() * torch.sigmoid(gate_p.reshape(extend_len, -1).float())).to(torch.bfloat16)
        torch.testing.assert_close(gated_p, expected_p, atol=2e-2, rtol=2e-2)

    # 6) CPU fallback path.
    backend_cpu, layer_cpu, _, _, _ = _make_qsa_backend(
        num_slots=256,
        max_reqs=8,
        max_ctx=32,
        kv_dtype=torch.float32,
        device="cpu",
    )
    q_c = torch.randn((2, layer_cpu.tp_q_head_num, layer_cpu.head_dim), dtype=torch.float32)
    gate_c = torch.randn_like(q_c)
    req_c = torch.tensor([1, 2], dtype=torch.int32)
    seq_c = torch.tensor([8, 8], dtype=torch.int32)
    idx_c = torch.arange(8, dtype=torch.int32).unsqueeze(0).expand(2, -1).contiguous()
    backend_cpu.forward_metadata = QwenSparseAttnMetadata(
        sequence_lengths=seq_c,
        token_to_batch_idx=torch.tensor([0, 1], dtype=torch.int32),
        token_slot_table=backend_cpu.req_to_token[req_c.long(), :8],
        indexer_metadata=None,
        row_req_pool_indices=req_c,
        is_cuda_graph=False,
    )
    fb_c = SimpleNamespace(forward_mode=ForwardMode.DECODE, req_pool_indices=req_c)
    unfused_c = backend_cpu._forward_paged_attention(q_c, layer_cpu, fb_c, idx_c, output_gate=None)
    gated_c = backend_cpu._forward_paged_attention(q_c, layer_cpu, fb_c, idx_c, output_gate=gate_c)
    torch.testing.assert_close(
        gated_c, unfused_c * torch.sigmoid(gate_c.reshape(2, -1)), atol=1e-5, rtol=1e-5
    )


def test_copy_indexed_ssm_rows_and_scatter_conv_guards():
    """Verify device checks on index tensors and aliasing protection in
    copy_indexed_ssm_rows / scatter_conv_track_rows."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        copy_indexed_ssm_rows,
        scatter_conv_track_rows,
    )

    device = "cuda"
    src = torch.randn((16, 4, 64, 64), dtype=torch.bfloat16, device=device)
    dst = torch.zeros_like(src)
    s_idx = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device=device)
    d_idx = torch.tensor([1, 3, 4, 9], dtype=torch.int64, device=device)

    # CPU index tensors must be rejected cleanly without launching Triton.
    assert not copy_indexed_ssm_rows(src, s_idx.cpu(), dst, d_idx)
    assert not copy_indexed_ssm_rows(src, s_idx, dst, d_idx.cpu())

    # Distinct tensors succeed and match PyTorch bit-for-bit.
    assert copy_indexed_ssm_rows(src, s_idx, dst, d_idx)
    ref = torch.zeros_like(src)
    ref[d_idx] = src[s_idx]
    assert torch.equal(dst, ref)

    # Aliased tensor with overlapping indices MUST return False by default so
    # callers fall back to PyTorch's atomic gather-then-scatter (no data race).
    aliased = src.clone()
    overlap_src = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    overlap_dst = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
    assert not copy_indexed_ssm_rows(aliased, overlap_src, aliased, overlap_dst)
    # Disjoint rows on aliased tensor succeed when explicitly opted in.
    disjoint_src = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
    disjoint_dst = torch.tensor([10, 11, 12], dtype=torch.int64, device=device)
    expected_aliased = aliased.clone()
    expected_aliased[disjoint_dst] = expected_aliased[disjoint_src]
    assert copy_indexed_ssm_rows(
        aliased, disjoint_src, aliased, disjoint_dst, assume_disjoint_rows=True
    )
    assert torch.equal(aliased, expected_aliased)

    # scatter_conv_track_rows rejects CPU indices and matches PyTorch on GPU.
    mixed_qkv = torch.randn((32, 128), dtype=torch.bfloat16, device=device)
    conv_states = torch.zeros((16, 128, 4), dtype=torch.bfloat16, device=device)
    t_idx = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=torch.int64, device=device)
    d_slots = torch.tensor([5, 9], dtype=torch.int64, device=device)
    assert not scatter_conv_track_rows(mixed_qkv, t_idx.cpu(), conv_states, d_slots)
    assert not scatter_conv_track_rows(mixed_qkv, t_idx, conv_states, d_slots.cpu())
    assert scatter_conv_track_rows(mixed_qkv, t_idx, conv_states, d_slots)
    ref_conv = torch.zeros_like(conv_states)
    ref_conv[d_slots] = mixed_qkv[t_idx].transpose(1, 2)
    assert torch.equal(conv_states, ref_conv)


def test_ple_shared_scratch_pool_reused_and_capped():
    """Verify multiple PLE layers share one _PLEScratchPool (instead of N_layers
    permanent VRAM copies) and oversized batches bypass permanent retention."""
    from sglang.srt.models.qwen4_exp import _PLEScratchPool

    pool = _PLEScratchPool(max_retained_tokens=256)
    a1, b1, c1 = pool.get(64, 512, torch.bfloat16, torch.device("cuda"))
    a2, b2, c2 = pool.get(64, 512, torch.bfloat16, torch.device("cuda"))
    assert a1 is not None and a2 is not None
    assert a1.data_ptr() == a2.data_ptr()
    assert b1.data_ptr() == b2.data_ptr()
    assert c1.data_ptr() == c2.data_ptr()

    # Growing up to cap keeps one shared backing tuple.
    a3, b3, c3 = pool.get(128, 512, torch.bfloat16, torch.device("cuda"))
    assert a3.shape == (128, 512)
    ptr_128 = a3.data_ptr()

    # Oversized token count > cap returns (None, None, None) and does NOT grow
    # or replace the retained pool.
    over = pool.get(512, 512, torch.bfloat16, torch.device("cuda"))
    assert over == (None, None, None)
    assert pool._buffers[0].shape == (128, 512)
    assert pool._buffers[0].data_ptr() == ptr_128


def test_mrope_text_only_fast_path_matches_legacy_cpu():
    """Verify ForwardBatch._compute_mrope_positions text-only GPU fast path
    matches the legacy CPU loop bit-for-bit across ragged requests and does not
    intercept multimodal batches."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    device = torch.device("cuda")
    extend_lens = [5, 1, 9, 3]
    prefix_lens = [0, 12, 4, 100]
    positions = torch.cat(
        [
            torch.arange(p, p + e, dtype=torch.int64, device=device)
            for p, e in zip(prefix_lens, extend_lens)
        ]
    )
    batch = SimpleNamespace(
        batch_size=len(extend_lens),
        multimodal_inputs=[None] * len(extend_lens),
        extend_lens=extend_lens,
        prefix_lens=prefix_lens,
    )
    seq_lens_cpu = torch.tensor(
        [p + e for p, e in zip(prefix_lens, extend_lens)], dtype=torch.int64
    )
    runner = SimpleNamespace(device=device)

    fb_fast = object.__new__(ForwardBatch)
    fb_fast.forward_mode = ForwardMode.EXTEND
    fb_fast.seq_lens_cpu = seq_lens_cpu
    fb_fast.positions = positions
    fb_fast._compute_mrope_positions(runner, batch)

    # Force legacy CPU loop by setting positions=None.
    fb_legacy = object.__new__(ForwardBatch)
    fb_legacy.forward_mode = ForwardMode.EXTEND
    fb_legacy.seq_lens_cpu = seq_lens_cpu
    fb_legacy.positions = None
    fb_legacy._compute_mrope_positions(runner, batch)

    assert fb_fast.mrope_positions.shape == (3, sum(extend_lens))
    assert fb_fast.mrope_positions.dtype == torch.int64
    assert torch.equal(fb_fast.mrope_positions, fb_legacy.mrope_positions)

    # Multimodal batch must bypass the text-only fast path.
    mm_item = SimpleNamespace(
        mrope_positions=torch.arange(3 * 5, dtype=torch.int64).reshape(3, 5)
    )
    batch_mm = SimpleNamespace(
        batch_size=1,
        multimodal_inputs=[mm_item],
        extend_lens=[5],
        prefix_lens=[0],
        reqs=[SimpleNamespace(session=None)],
    )
    fb_mm = object.__new__(ForwardBatch)
    fb_mm.forward_mode = ForwardMode.EXTEND
    fb_mm.seq_lens_cpu = torch.tensor([5], dtype=torch.int64)
    fb_mm.positions = torch.arange(5, dtype=torch.int64, device=device)
    fb_mm._compute_mrope_positions(runner, batch_mm)
    assert torch.equal(fb_mm.mrope_positions.cpu(), mm_item.mrope_positions)


def test_qwen2_moe_dual_stream_skip_clone_when_not_inplace():
    """Verify Qwen2MoeSparseMoeBlock.forward_normal_dual_stream skips clone()
    when moe_runner_config.inplace=False, and clones when inplace=True so
    in-place router mutation cannot corrupt shared_expert input."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    device = torch.device("cuda")
    x = torch.randn((32, 64), dtype=torch.bfloat16, device=device)
    x_orig = x.clone()

    for inplace in (False, True):
        block = object.__new__(Qwen2MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.alt_stream = torch.cuda.Stream()
        block.shared_expert = object()
        block.experts = SimpleNamespace(
            moe_runner_config=SimpleNamespace(inplace=inplace)
        )
        seen = {}

        def _shared(inp, apply_gate=True):
            seen["ptr"] = inp.data_ptr()
            return inp * 2.0

        def _router(inp):
            if inplace:
                inp.fill_(-99.0)
                return torch.ones_like(inp)
            return inp * 3.0

        block._forward_shared_experts = _shared
        block._forward_router_experts = _router

        inp = x_orig.clone()
        r_out, s_out = block.forward_normal_dual_stream(inp, use_fused_gate=False)
        torch.cuda.synchronize()
        if inplace:
            assert seen["ptr"] != inp.data_ptr()
            torch.testing.assert_close(s_out, x_orig * 2.0)
        else:
            assert seen["ptr"] == inp.data_ptr()
            torch.testing.assert_close(s_out, x_orig * 2.0)
            torch.testing.assert_close(r_out, x_orig * 3.0)


def test_tokenizer_manager_parallelism_setdefault(monkeypatch):
    """Verify TokenizerManager uses setdefault for TOKENIZERS_PARALLELISM:
    keeps default 'false' when unset, and preserves explicit 'true'."""
    import ast
    from pathlib import Path

    tm_path = Path(
        __import__(
            "sglang.srt.managers.tokenizer_manager", fromlist=["TokenizerManager"]
        ).__file__
    )
    src = tm_path.read_text()
    assert 'os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")' in src
    assert 'os.environ["TOKENIZERS_PARALLELISM"] = "false"' not in src

    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"


def test_mtp_split_last_token_draft_extend_v2_layout_and_attention_matches_eager():
    """Verify reusing DRAFT_EXTEND_V2 CUDA graph (captured_req_width=4 > 1)
    for MTP split-prefill last_token (extend_seq_lens=[1]*bs) on bs>1 (both
    unpadded bs=4 and padded bs=3 -> capture_bs=4) matches eager EXTEND on
    QSA metadata, attention output, select_index gather, and MTP shared
    sparse-index capture."""
    from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

    device = "cuda"
    captured_req_width = 4
    capture_bs = 4
    num_graph_rows = capture_bs * captured_req_width  # 16
    compress_ratio = 4
    token_topk = 64

    for raw_bs in (3, 4):
        backend_g, layer, k_buf, v_buf, req_to_token = _make_qsa_backend(
            num_slots=4096,
            max_reqs=16,
            max_ctx=256,
            compress_ratio=compress_ratio,
            token_topk=token_topk,
            cuda_graph_max_tokens=num_graph_rows,
            device=device,
        )
        # Share identical KV pool & req_to_token with eager backend.
        backend_e, _, _, _, _ = _make_qsa_backend(
            num_slots=4096,
            max_reqs=16,
            max_ctx=256,
            compress_ratio=compress_ratio,
            token_topk=token_topk,
            cuda_graph_max_tokens=num_graph_rows,
            device=device,
        )
        backend_e.token_to_kv_pool.layer_router._kv._k.copy_(k_buf)
        backend_e.token_to_kv_pool.layer_router._kv._v.copy_(v_buf)
        backend_e.req_to_token.copy_(req_to_token)

        expanded_width = token_topk + compress_ratio - 1
        share_g = QSAMTPSharedSparseIndices(
            layer_ids=[0],
            num_requests=16,
            token_topk=expanded_width,
            tail_width=4,
            device=torch.device(device),
        )
        share_e = QSAMTPSharedSparseIndices(
            layer_ids=[0],
            num_requests=16,
            token_topk=expanded_width,
            tail_width=4,
            device=torch.device(device),
        )
        backend_g.set_mtp_shared_sparse_indices(share_g)
        backend_e.set_mtp_shared_sparse_indices(share_e)

        # Build realistic last_batch inputs (1 token per request, ragged seq_lens).
        real_reqs = torch.tensor([2, 5, 9, 11][:raw_bs], dtype=torch.int64, device=device)
        real_seqs = torch.tensor([97, 132, 85, 160][:raw_bs], dtype=torch.int64, device=device)
        q_real = torch.randn((raw_bs, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
        gate_real = torch.randn_like(q_real)
        topk_real = torch.stack(
            [
                torch.arange(token_topk, dtype=torch.int32, device=device)
                for _ in range(raw_bs)
            ],
            dim=0,
        )
        # Padded graph buffers matching EAGLEDraftExtendCudaGraphRunner.execute().
        buf_reqs = torch.zeros(capture_bs, dtype=torch.int64, device=device)
        buf_reqs[:raw_bs] = real_reqs
        buf_seqs = torch.full((capture_bs,), 8, dtype=torch.int64, device=device)
        buf_seqs[:raw_bs] = real_seqs
        buf_ext = torch.full((capture_bs,), captured_req_width, dtype=torch.int32, device=device)
        buf_ext[:raw_bs] = 1
        buf_accept = torch.full((capture_bs,), captured_req_width, dtype=torch.int32, device=device)
        buf_accept[:raw_bs] = 1
        buf_q = torch.zeros((num_graph_rows, layer.tp_q_head_num, layer.head_dim), dtype=torch.bfloat16, device=device)
        buf_q[:raw_bs] = q_real
        buf_gate = torch.zeros_like(buf_q)
        buf_gate[:raw_bs] = gate_real
        buf_topk = torch.zeros((num_graph_rows, token_topk), dtype=torch.int32, device=device)
        buf_topk[:raw_bs] = topk_real
        select_index = torch.arange(raw_bs, dtype=torch.int64, device=device)

        # Graph metadata buffers rebuilt by mode=2 (DRAFT_EXTEND_V2).
        max_pages = math.ceil(math.ceil(256 / compress_ratio) / 16)
        idx_meta_g = QSAIndexerMetadata(
            sequence_lengths=torch.zeros(num_graph_rows, dtype=torch.int32, device=device),
            token_to_batch_idx=torch.arange(num_graph_rows, dtype=torch.int32, device=device),
            token_slot_table=torch.zeros((num_graph_rows, 1), dtype=torch.int32, device=device),
            out_cache_loc=torch.zeros(num_graph_rows, dtype=torch.int64, device=device),
            token_to_kv_pool=backend_g.token_to_kv_pool,
            compress_ratio=compress_ratio,
            block_topk=token_topk // compress_ratio,
            req_pool_indices=torch.zeros(num_graph_rows, dtype=torch.int32, device=device),
            is_cuda_graph=True,
            graph_write_locs=torch.zeros(num_graph_rows, dtype=torch.int32, device=device),
            graph_compressed_page_table=torch.zeros((num_graph_rows, max_pages), dtype=torch.int32, device=device),
            graph_compressed_lengths=torch.zeros(num_graph_rows, dtype=torch.int32, device=device),
            graph_prefix_lengths=torch.zeros(num_graph_rows, dtype=torch.int32, device=device),
            decode_logical_positions=torch.zeros(num_graph_rows, dtype=torch.int32, device=device),
            pending_ring_slots=torch.zeros(num_graph_rows, dtype=torch.int64, device=device),
            graph_ring_group_locs=torch.zeros((num_graph_rows, compress_ratio), dtype=torch.int32, device=device),
        )
        meta_g = QwenSparseAttnMetadata(
            sequence_lengths=idx_meta_g.sequence_lengths,
            token_to_batch_idx=idx_meta_g.token_to_batch_idx,
            token_slot_table=idx_meta_g.token_slot_table,
            indexer_metadata=idx_meta_g,
            row_req_pool_indices=idx_meta_g.req_pool_indices,
            is_cuda_graph=True,
        )
        launch_graph_metadata(
            mode=2,
            bs=capture_bs,
            num_rows=num_graph_rows,
            seq_lens=buf_seqs,
            req_pool_indices=buf_reqs,
            extend_lens=buf_ext,
            extend_len=0,
            num_padding=0,
            metadata=meta_g,
            req_to_token=req_to_token,
            pool=backend_g.token_to_kv_pool,
        )
        backend_g.forward_metadata = meta_g
        spec_g = SimpleNamespace(
            num_tokens_per_req=captured_req_width,
            num_front_tokens=0,
            num_accept_tokens=buf_accept,
            extend_seq_lens_tensor=buf_ext,
        )
        fb_g = SimpleNamespace(
            batch_size=capture_bs,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            _original_forward_mode=None,
            req_pool_indices=buf_reqs,
            extend_seq_lens=buf_ext,
            spec_info=spec_g,
            attn_cp_metadata=None,
        )

        # Capture & replay DRAFT_EXTEND_V2 graph attention + MTP index capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            backend_g._forward_paged_attention(buf_q, layer, fb_g, buf_topk, output_gate=buf_gate)
            backend_g.capture_mtp_sparse_indices(buf_topk, fb_g, 0, idx_meta_g)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out_all = backend_g._forward_paged_attention(buf_q, layer, fb_g, buf_topk, output_gate=buf_gate)
            backend_g.capture_mtp_sparse_indices(buf_topk, fb_g, 0, idx_meta_g)
        graph.replay()
        torch.cuda.synchronize()
        g_out = g_out_all[select_index]

        # Eager EXTEND reference (1 token per request, no padding).
        idx_meta_e = QSAIndexerMetadata(
            sequence_lengths=real_seqs.to(torch.int32),
            token_to_batch_idx=torch.arange(raw_bs, dtype=torch.int32, device=device),
            token_slot_table=req_to_token[real_reqs],
            out_cache_loc=torch.zeros(raw_bs, dtype=torch.int64, device=device),
            token_to_kv_pool=backend_e.token_to_kv_pool,
            compress_ratio=compress_ratio,
            block_topk=token_topk // compress_ratio,
            req_pool_indices=real_reqs.to(torch.int32),
            is_cuda_graph=False,
        )
        meta_e = QwenSparseAttnMetadata(
            sequence_lengths=real_seqs.to(torch.int32),
            token_to_batch_idx=torch.arange(raw_bs, dtype=torch.int32, device=device),
            token_slot_table=req_to_token[real_reqs],
            indexer_metadata=idx_meta_e,
            row_req_pool_indices=real_reqs.to(torch.int32),
            is_cuda_graph=False,
        )
        backend_e.forward_metadata = meta_e
        fb_e = SimpleNamespace(
            batch_size=raw_bs,
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            req_pool_indices=real_reqs,
            extend_seq_lens=torch.ones(raw_bs, dtype=torch.int32, device=device),
            extend_seq_lens_cpu=[1] * raw_bs,
            seq_lens=real_seqs,
            seq_lens_cpu=real_seqs.cpu().tolist(),
            spec_info=SimpleNamespace(num_front_tokens=0, num_accept_tokens=None),
            attn_cp_metadata=None,
        )
        # Eager paged attention on exact same FP8 pool & topk.
        e_out = backend_e._forward_paged_attention(q_real, layer, fb_e, topk_real, output_gate=gate_real)
        backend_e.capture_mtp_sparse_indices(topk_real, fb_e, 0, idx_meta_e)

        # Verify real-row metadata, attention outputs, and MTP shared indices match!
        assert torch.equal(meta_g.sequence_lengths[:raw_bs], real_seqs.to(torch.int32))
        assert torch.equal(meta_g.row_req_pool_indices[:raw_bs], real_reqs.to(torch.int32))
        torch.testing.assert_close(g_out, e_out, atol=1e-2, rtol=1e-2)
        for r in real_reqs.tolist():
            assert torch.equal(share_g.indices[0, r, :token_topk], share_e.indices[0, r, :token_topk])
            assert int(share_g.captured_len[0, r]) == int(share_e.captured_len[0, r])
        # Slot 0 must remain untouched (padding routed to trash_row).
        assert int(share_g.captured_len[0, 0]) == 1


def test_qsa_graph_metadata_triton_single_compile_across_bs_and_alignment():
    """Verify QSA graph metadata Triton kernels compile exactly once across
    varying batch sizes, padding counts, and unaligned pointer slices."""
    import triton

    from sglang.srt.layers.attention.qsa.graph_metadata import (
        _qsa_draft_graph_metadata_kernel,
        _qsa_graph_layout_kernel,
        _qsa_graph_row_metadata_kernel,
    )

    device = "cuda"
    max_bs, max_rows, max_pages, ratio, full_page = 64, 256, 64, 4, 64
    req_to_token = torch.arange(
        128 * 4096, dtype=torch.int32, device=device
    ).reshape(128, 4096)

    def _bufs():
        return (
            torch.zeros(max_rows, dtype=torch.int32, device=device),
            torch.zeros(max_rows, dtype=torch.int32, device=device),
            torch.zeros(max_rows, dtype=torch.int32, device=device),
            torch.zeros(max_rows, dtype=torch.int32, device=device),
            torch.zeros((max_rows, max_pages), dtype=torch.int32, device=device),
            torch.zeros(max_rows, dtype=torch.int32, device=device),
            torch.zeros(max_rows, dtype=torch.int64, device=device),
            torch.zeros((max_rows, ratio), dtype=torch.int32, device=device),
            torch.zeros(max_rows, dtype=torch.int32, device=device),
        )

    b0 = _bufs()
    draft_bufs = (_bufs(), _bufs())
    seq_base = torch.full((max_bs + 5,), 128, dtype=torch.int64, device=device)
    req_base = torch.arange(1, max_bs + 6, dtype=torch.int64, device=device)
    ext_base = torch.full((max_bs + 5,), 2, dtype=torch.int32, device=device)

    for fn in (
        _qsa_graph_layout_kernel,
        _qsa_graph_row_metadata_kernel,
        _qsa_draft_graph_metadata_kernel,
    ):
        fn.device_caches.clear()

    for shift in (0, 1):
        for bs, pad in ((64, 0), (64, 1), (48, 3), (32, 0), (16, 1), (8, 0), (1, 0)):
            sl = seq_base[shift : shift + bs]
            rp = req_base[shift : shift + bs]
            ex = ext_base[shift : shift + bs]
            nr = bs * 4
            _qsa_graph_layout_kernel[(bs + 1,)](
                sl, rp, ex, b0[0], b0[8], b0[1], bs, nr, pad, num_warps=1
            )
            _qsa_graph_row_metadata_kernel[(nr, triton.cdiv(max_pages, 128))](
                b0[0],
                b0[1],
                b0[2],
                b0[3],
                b0[4],
                b0[5],
                b0[6],
                b0[7],
                req_to_token,
                req_to_token.stride(0),
                max_pages,
                RATIO=ratio,
                FULL_PAGE=full_page,
                PAGE_BLOCK=128,
                seq_lens_ptr=sl,
                req_pool_ptr=rp,
                row_prefix_lens_ptr=b0[8],
                bs=bs,
                num_padding=pad,
                extend_len=4,
                seq_offset=0,
                MODE=2,
                num_warps=1,
            )
            _qsa_draft_graph_metadata_kernel[(bs, triton.cdiv(max_pages, 128), 2)](
                draft_bufs,
                sl,
                rp,
                req_to_token,
                req_to_token.stride(0),
                bs,
                pad,
                max_pages,
                ratio,
                full_page,
                num_warps=1,
            )
    torch.cuda.synchronize()
    assert len(_qsa_graph_layout_kernel.device_caches[0][0]) == 1
    assert len(_qsa_graph_row_metadata_kernel.device_caches[0][0]) == 1
    assert len(_qsa_draft_graph_metadata_kernel.device_caches[0][0]) == 1
