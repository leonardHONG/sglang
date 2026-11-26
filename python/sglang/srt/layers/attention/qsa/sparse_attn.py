"""Validated sparse GQA operators migrated from the QSA reference branch."""

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
# SM90+ (Hopper/Blackwell): for 1-byte FP8 K/V caches, 3 pipeline stages fit in
# shared memory at high occupancy and hide HBM gather latency (-12.7% on B300).
# For 2-byte BF16/FP16 K/V, 3 stages double shared-memory pressure and regress
# occupancy (+20-35% slower), so keep 2 stages.
_SM90_PLUS_FP8_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 3)),
]
# SM90+ FP8 int16-pair PTX kernel: 64-token tiles with 4 warps and deep
# cp.async pipeline hide scattered FP8 HBM gather latency on Hopper/Blackwell.
_SM90_PLUS_FP8_PAIRS_CONFIGS = [
    (64, (32, 4, 2)),
    (1024, (32, 4, 3)),
    (float("inf"), (64, 4, 5)),
]


@functools.lru_cache(maxsize=1)
def _supports_fp8_pairs_ptx() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0) >= (9, 0)
    )


def _get_fp8_pairs_config(total_q: int):
    return next(cfg for limit, cfg in _SM90_PLUS_FP8_PAIRS_CONFIGS if total_q <= limit)


def _normalize_output_gate(
    output_gate: Optional[torch.Tensor],
    total_q: int,
    num_q_heads: int,
    head_dim: int,
) -> Optional[torch.Tensor]:
    if output_gate is None:
        return None
    gate = output_gate[:total_q]
    if gate.ndim == 2:
        gate = gate.view(total_q, num_q_heads, head_dim)
    return gate


@triton.jit
def _e4m3_pairs_to_fp16(pairs, HIGH: tl.constexpr):
    """Unpack one byte (byte 0 = dim 2i, byte 1 = dim 2i+1) of each int16 pair
    of FP8-E4M3 codes into FP16 using pack=2 inline PTX (4 bytes/insn)."""
    if HIGH:
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 t;
                .reg .b16 lo, hi;
                prmt.b32 t, $1, 0, 0x3120;
                mov.b32 {lo, hi}, t;
                cvt.rn.f16x2.e4m3x2 $0, hi;
            }
            """,
            constraints="=r,r",
            args=[pairs],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
    else:
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 t;
                .reg .b16 lo, hi;
                prmt.b32 t, $1, 0, 0x3120;
                mov.b32 {lo, hi}, t;
                cvt.rn.f16x2.e4m3x2 $0, lo;
            }
            """,
            constraints="=r,r",
            args=[pairs],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )


@functools.lru_cache(maxsize=2)
def _prefill_config_table(is_fp8: bool = False):
    if not torch.cuda.is_available():
        return _L20_CONFIGS
    device_name = torch.cuda.get_device_name(0)
    if "H20" in device_name and "H200" not in device_name:
        return _H20_CONFIGS
    if (
        is_fp8
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0) >= (9, 0)
    ):
        return _SM90_PLUS_FP8_CONFIGS
    return _L20_CONFIGS


def _get_best_config(total_q: int, *, is_fp8: bool = False):
    if is_batch_invariant_mode_enabled():
        return (32, 4, 2)
    table = _prefill_config_table(is_fp8=is_fp8)
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    gate,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sg_m: tl.constexpr,
    sg_h: tl.constexpr,
    sg_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    if HAS_GATE:
        g_vals = tl.load(
            gate
            + query * sg_m
            + (head_start + offs_h[:, None]) * sg_h
            + offs_d[None, :] * sg_d,
            mask=(offs_h < GROUP_SIZE)[:, None],
            other=0.0,
        ).to(tl.float32)
        output = output * tl.sigmoid(g_vals)
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(
    q,
    k,
    v,
    max_seqlen_k,
    indices,
    cu_seqlens,
    scale,
    *,
    out: Optional[torch.Tensor] = None,
    output_gate: Optional[torch.Tensor] = None,
):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q, is_fp8=k.element_size() == 1)
    if out is None:
        out = torch.empty_like(q)
    gate_3d = _normalize_output_gate(output_gate, total_q, num_q_heads, head_dim)
    has_gate = gate_3d is not None
    gate_arg = gate_3d if has_gate else q
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        gate_arg,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        gate_arg.stride(0) if has_gate else 0,
        gate_arg.stride(1) if has_gate else 0,
        gate_arg.stride(2) if has_gate else 0,
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        HAS_GATE=has_gate,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    gate,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sg_m: tl.constexpr,
    sg_h: tl.constexpr,
    sg_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOT_INDICES: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    """One query row x one KV head group over its ``topk`` selected tokens.

    ``indices`` hold positions relative to row ``cu_k[batch]`` of ``k``/``v``
    (a packed context), or, with ``SLOT_INDICES``, rows of ``k``/``v`` itself
    (KV-pool slots; ``cu_k`` is unused)."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    if SLOT_INDICES:
        k_start = 0
    else:
        k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        if SLOT_INDICES:
            # 64-bit: slot * sk_n overflows int32 on a pool of a few M tokens.
            token = token.to(tl.int64)
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # Direct callers and the KV pool can be FP8; packing uses Q's dtype.
        # QSA stores unscaled K/V, so a plain cast supports both callers.
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    if HAS_GATE:
        g_vals = tl.load(
            gate
            + query * sg_m
            + (group * GROUP_SIZE + offs_h[:, None]) * sg_h
            + offs_d[None, :] * sg_d,
            mask=(offs_h < GROUP_SIZE)[:, None],
            other=0.0,
        ).to(tl.float32)
        output = output * tl.sigmoid(g_vals)
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


@triton.jit
def _sparse_gqa_chunk_prefill_fp8_pairs(
    q,
    k_pairs_ptr,
    v_pairs_ptr,
    out,
    gate,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sg_m: tl.constexpr,
    sg_h: tl.constexpr,
    sg_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SLOT_INDICES: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ALIGNED_8: tl.constexpr,
):
    """SM90+ FP8-E4M3 sparse prefill using 16B-aligned int16-pair loads,
    pack=2 inline PTX cvt.rn.f16x2.e4m3x2 unpacking, and pipelined K/V gathers."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    if SLOT_INDICES:
        k_start = 0
    else:
        k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    num_tiles = tl.cdiv(row_topk, BLOCK_N)

    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_half_d = tl.arange(0, HALF_DIM)
    head_start = group * GROUP_SIZE
    head_mask = (offs_h < GROUP_SIZE)[:, None]

    q_pair_base = (
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + (2 * offs_half_d[None, :]) * sq_d
    )
    q_even = tl.load(q_pair_base, mask=head_mask, other=0.0)
    q_odd = tl.load(q_pair_base + sq_d, mask=head_mask, other=0.0)
    q_dtype = q_even.dtype
    score_scale = scale * 1.4426950408889634

    k_base = k_pairs_ptr + k_start * sk_n + group * sk_h
    v_base = v_pairs_ptr + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g

    max_value = tl.full([BLOCK_M], -1.0e20, tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    acc_even = tl.zeros([BLOCK_M, HALF_DIM], tl.float32)
    acc_odd = tl.zeros([BLOCK_M, HALF_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for tile in tl.range(0, num_tiles, num_stages=NUM_STAGES):
        current = tile * BLOCK_N + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        safe_token = tl.maximum(token, 0).to(tl.int64)

        k_tok_off = safe_token * sk_n
        v_tok_off = safe_token * sv_n
        if ALIGNED_8:
            k_tok_off = tl.multiple_of(k_tok_off, 8)
            v_tok_off = tl.multiple_of(v_tok_off, 8)
        key_pairs = tl.trans(
            tl.load(
                k_base + k_tok_off[:, None] + offs_half_d[None, :],
                mask=valid[:, None],
                other=0,
            )
        )
        value_pairs = tl.load(
            v_base + v_tok_off[:, None] + offs_half_d[None, :],
            mask=valid[:, None],
            other=0,
        )

        scores = tl.dot(
            q_even, _e4m3_pairs_to_fp16(key_pairs, False).to(q_dtype)
        ) + tl.dot(q_odd, _e4m3_pairs_to_fp16(key_pairs, True).to(q_dtype))
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)

        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        probs_fp16 = probabilities.to(tl.float16)

        acc_even = tl.dot(
            probs_fp16,
            _e4m3_pairs_to_fp16(value_pairs, False),
            acc=acc_even * alpha[:, None],
        )
        acc_odd = tl.dot(
            probs_fp16,
            _e4m3_pairs_to_fp16(value_pairs, True),
            acc=acc_odd * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    inv_norm = 1.0 / tl.maximum(normalizer, 1.0e-20)
    out_even = tl.where(has_values[:, None], acc_even * inv_norm[:, None], 0.0)
    out_odd = tl.where(has_values[:, None], acc_odd * inv_norm[:, None], 0.0)
    output = tl.interleave(out_even, out_odd)
    if HAS_GATE:
        g_vals = tl.load(
            gate
            + query * sg_m
            + (head_start + offs_h[:, None]) * sg_h
            + offs_d[None, :] * sg_d,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        output = output * tl.sigmoid(g_vals)
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output.to(out.dtype.element_ty),
        mask=head_mask,
    )


# SR0 varies per batch for the KV-shard scratch slot table; do not recompile on it.
@triton.jit(do_not_specialize=["TOTAL_K", "BS", "SR0"])
def _pack_qsa_prefill_kv(
    k,
    v,
    packed_k,
    packed_v,
    req_to_token,
    req_indices,
    cu_k,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SR0,
    SR1: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    TOTAL_K,
    BS,
    COMPACT: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // (HEADS * DIM)
    if COMPACT:
        valid = token < TOTAL_K
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = lo + BS
        steps = BS
        while steps > 0:
            mid = (lo + hi + 1) // 2
            start = tl.load(cu_k + mid).to(tl.int64)
            after = token >= start
            lo = tl.where(after, mid, lo)
            hi = tl.where(after, hi, mid - 1)
            steps = steps // 2
        start = tl.load(cu_k + lo, valid, 0).to(tl.int64)
        req = tl.load(req_indices + lo, valid, 0).to(tl.int64)
        token = token - start
        dst = offsets
    else:
        batch = tl.program_id(1)
        start = tl.load(cu_k + batch).to(tl.int64)
        end = tl.load(cu_k + batch + 1).to(tl.int64)
        valid = token < end - start
        req = tl.load(req_indices + batch).to(tl.int64)
        dst = start * HEADS * DIM + offsets
    head = offsets // DIM % HEADS
    dim = offsets % DIM
    slot = tl.load(req_to_token + req * SR0 + token * SR1, valid, 0).to(tl.int64)
    keys = tl.load(k + slot * SK0 + head * SK1 + dim * SK2, valid, 0.0)
    values = tl.load(v + slot * SV0 + head * SV1 + dim * SV2, valid, 0.0)
    tl.store(packed_k + dst, keys, valid)
    tl.store(packed_v + dst, values, valid)


def pack_qsa_prefill_kv(
    *, k, v, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype=None
):
    heads, dim = k.shape[1:]
    packed_k = torch.empty(
        (total_k, heads, dim), dtype=output_dtype or k.dtype, device=k.device
    )
    packed_v = torch.empty(
        (total_k, heads, dim), dtype=output_dtype or v.dtype, device=v.device
    )
    batch_size = req_indices.numel()
    blocks_per_request = triton.cdiv(max_k * heads * dim, 1024)
    compact = (
        max_k * batch_size > 2 * total_k and blocks_per_request * batch_size >= 4096
    )
    grid = (
        (triton.cdiv(total_k * heads * dim, 1024),)
        if compact
        else (blocks_per_request, batch_size)
    )
    _pack_qsa_prefill_kv[grid](
        k,
        v,
        packed_k,
        packed_v,
        req_to_token,
        req_indices,
        cu_k,
        *k.stride(),
        *v.stride(),
        *req_to_token.stride(),
        HEADS=heads,
        DIM=dim,
        BLOCK=1024,
        TOTAL_K=total_k,
        BS=batch_size,
        COMPACT=compact,
    )
    return packed_k, packed_v


def _can_use_fp8_pairs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[bool, bool, bool]:
    """Returns (use_fp8_pairs_kernel, packed_int16_pairs, aligned_8)."""
    head_dim = q.shape[2]
    if not (
        _supports_fp8_pairs_ptx()
        and k.dtype == torch.float8_e4m3fn
        and v.dtype == torch.float8_e4m3fn
        and q.dtype in (torch.bfloat16, torch.float16)
        and head_dim >= 32
        and head_dim % 4 == 0
        and not is_batch_invariant_mode_enabled()
    ):
        return False, False, False
    packed = (
        k.stride(-1) == 1
        and v.stride(-1) == 1
        and k.stride(0) % 2 == 0
        and k.stride(1) % 2 == 0
        and v.stride(0) % 2 == 0
        and v.stride(1) % 2 == 0
        and k.storage_offset() % 2 == 0
        and v.storage_offset() % 2 == 0
    )
    aligned_8 = (
        packed
        and k.stride(0) % 16 == 0
        and k.stride(1) % 16 == 0
        and v.stride(0) % 16 == 0
        and v.stride(1) % 16 == 0
        and k.data_ptr() % 16 == 0
        and v.data_ptr() % 16 == 0
    )
    return True, packed, aligned_8


def _launch_chunk_prefill_fp8_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    gate_arg: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: float,
    *,
    max_q: int,
    num_entries: int,
    num_kv_heads: int,
    group_size: int,
    block_m: int,
    head_dim: int,
    has_gate: bool,
    slot_indices: bool,
    packed_pairs: bool,
    aligned_8: bool,
) -> None:
    if not packed_pairs:
        k, v = k.contiguous(), v.contiguous()
        _, _, aligned_8 = _can_use_fp8_pairs(q, k, v)
    k_arg = k.view(torch.int16)
    v_arg = v.view(torch.int16)
    block_n, warps, stages = _get_fp8_pairs_config(q.shape[0])
    _sparse_gqa_chunk_prefill_fp8_pairs[(max_q, num_entries * num_kv_heads)](
        q,
        k_arg,
        v_arg,
        out,
        gate_arg,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_arg.stride(0),
        k_arg.stride(1),
        k_arg.stride(2),
        v_arg.stride(0),
        v_arg.stride(1),
        v_arg.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        gate_arg.stride(0) if has_gate else 0,
        gate_arg.stride(1) if has_gate else 0,
        gate_arg.stride(2) if has_gate else 0,
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        NUM_STAGES=stages,
        SLOT_INDICES=slot_indices,
        HAS_GATE=has_gate,
        ALIGNED_8=aligned_8,
        num_warps=warps,
        num_stages=stages,
    )


def sparse_gqa_fwd_interface_triton_ck(
    q,
    k,
    v,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    max_q: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
    output_gate: Optional[torch.Tensor] = None,
):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    if max_q is None:
        # Host sync; callers that know the longest entry pass it instead.
        max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    if out is None:
        out = torch.empty_like(q)
    gate_3d = _normalize_output_gate(output_gate, total_q, num_q_heads, head_dim)
    has_gate = gate_3d is not None
    gate_arg = gate_3d if has_gate else q
    use_fp8_pairs, packed_pairs, aligned_8 = _can_use_fp8_pairs(q, k, v)
    if use_fp8_pairs:
        _launch_chunk_prefill_fp8_pairs(
            q,
            k,
            v,
            out,
            gate_arg,
            indices,
            cu_q,
            cu_k,
            kv_lens,
            scale,
            max_q=max_q,
            num_entries=cu_q.shape[0] - 1,
            num_kv_heads=num_kv_heads,
            group_size=group_size,
            block_m=block_m,
            head_dim=head_dim,
            has_gate=has_gate,
            slot_indices=False,
            packed_pairs=packed_pairs,
            aligned_8=aligned_8,
        )
        return out
    block_n, warps, stages = _get_best_config(total_q, is_fp8=k.element_size() == 1)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        gate_arg,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        gate_arg.stride(0) if has_gate else 0,
        gate_arg.stride(1) if has_gate else 0,
        gate_arg.stride(2) if has_gate else 0,
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        SLOT_INDICES=False,
        HAS_GATE=has_gate,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def sparse_gqa_packed_decode_triton(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    """Run one packed sparse-attention row per request without a host sync.

    Reuses the chunk-prefill kernel at a fixed query length of one, so graph
    capture never hits the ``.item()`` that the general interface needs to
    derive ``max_q``.
    """

    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q, is_fp8=k.element_size() == 1)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(1, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        q,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        0,
        0,
        0,
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        SLOT_INDICES=False,
        HAS_GATE=False,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit(do_not_specialize=["slot_rows", "st_m"])
def _resolve_prefill_slots(
    indices,
    slots,
    cu_q,
    slot_table,
    slot_rows,
    topk,
    si_m,
    ss_m,
    st_m,
    BLOCK: tl.constexpr,
):
    """``slots[row, j] = slot_table[entry % slot_rows, indices[row, j]]``, -1
    where the index is; one program per query row of an entry."""
    entry = tl.program_id(1)
    q_start = tl.load(cu_q + entry)
    row = q_start + tl.program_id(0)
    if row >= tl.load(cu_q + entry + 1):
        return
    row = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    token = tl.load(indices + row * si_m + cols, mask=cols < topk, other=-1)
    valid = token >= 0
    table_row = slot_table + (entry % slot_rows).to(tl.int64) * st_m
    slot = tl.load(table_row + token, mask=valid, other=-1)
    tl.store(slots + row * ss_m + cols, slot, mask=cols < topk)


def sparse_gqa_paged_prefill_triton(
    q,
    k_cache,
    v_cache,
    slot_table,
    indices,
    cu_q,
    kv_lens,
    scale,
    *,
    max_q,
    out: Optional[torch.Tensor] = None,
    output_gate: Optional[torch.Tensor] = None,
):
    """Chunk-prefill sparse attention that reads K/V in place from the KV pool.

    Same entry geometry as ``sparse_gqa_fwd_interface_triton_ck`` (``cu_q`` and
    ``kv_lens`` per entry, ``indices`` are logical positions of the entry's
    request), but no packed copy of each request's context: entry ``e``'s
    position ``t`` is read from pool slot
    ``slot_table[e % slot_table.shape[0], t]``. Prefill CP passes two entries
    per request (zigzag prev blocks, then next blocks) over one table row per
    request. ``max_q`` is the longest entry's query count, from host lengths.
    The selected tokens, the current chunk's included, must already be in the
    pool.

    The slots are resolved in a pass of their own: looking them up inside the
    attention loop puts a second dependent load in front of every K/V load and
    made the kernel 1.5x slower."""
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    num_entries = cu_q.shape[0] - 1
    if out is None:
        out = torch.empty_like(q)
    if total_q == 0 or num_entries == 0:
        return out
    indices = indices[:total_q]
    topk = indices.shape[-1]
    assert indices.ndim == 2 and indices.stride(-1) == 1
    assert slot_table.stride(-1) == 1
    slots = torch.empty((total_q, topk), dtype=slot_table.dtype, device=indices.device)
    _resolve_prefill_slots[(max_q, num_entries)](
        indices,
        slots,
        cu_q,
        slot_table,
        slot_table.shape[0],
        topk,
        indices.stride(0),
        slots.stride(0),
        slot_table.stride(0),
        BLOCK=triton.next_power_of_2(topk),
        num_warps=4,
    )
    block_m = max(16, triton.next_power_of_2(group_size))
    gate_3d = _normalize_output_gate(output_gate, total_q, num_q_heads, head_dim)
    has_gate = gate_3d is not None
    gate_arg = gate_3d if has_gate else q

    use_fp8_pairs, packed_pairs, aligned_8 = _can_use_fp8_pairs(q, k_cache, v_cache)
    if use_fp8_pairs:
        _launch_chunk_prefill_fp8_pairs(
            q,
            k_cache,
            v_cache,
            out,
            gate_arg,
            slots,
            cu_q,
            cu_q,
            kv_lens,
            scale,
            max_q=max_q,
            num_entries=num_entries,
            num_kv_heads=num_kv_heads,
            group_size=group_size,
            block_m=block_m,
            head_dim=head_dim,
            has_gate=has_gate,
            slot_indices=True,
            packed_pairs=packed_pairs,
            aligned_8=aligned_8,
        )
        return out

    block_n, warps, stages = _get_best_config(
        total_q, is_fp8=k_cache.element_size() == 1
    )
    _sparse_gqa_chunk_prefill[(max_q, num_entries * num_kv_heads)](
        q,
        k_cache,
        v_cache,
        out,
        gate_arg,
        slots,
        cu_q,
        cu_q,
        kv_lens,
        scale,
        topk,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        gate_arg.stride(0) if has_gate else 0,
        gate_arg.stride(1) if has_gate else 0,
        gate_arg.stride(2) if has_gate else 0,
        slots.stride(0),
        0,
        slots.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        SLOT_INDICES=True,
        HAS_GATE=has_gate,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # One request per program: Triton caps a tile at 1M elements,
    # which [next_pow2(topk), next_pow2(batch)] exceeds at topk=2051, batch=512.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # 64-bit element offsets: slot * heads * dim exceeds int32 once the pool holds
    # more than 2^31 / (heads * dim) tokens (~4.2M for 2 x 256), which an FP8 pool
    # on one GPU does reach.
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole pages,
        # so every slot in [valid_count, pad_cols) must hold zeros, never stale bytes.
        # `valid_count` here is the row's page-aligned stride, not its valid count, so
        # the store covers the full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    # Dequantize while gathering: the scratch is allocated in the query dtype, so an
    # FP8 pool is read as fp8 and stored as bf16. The QSA backend writes the pool
    # without per-tensor k/v scales (see set_kv_buffer calls in
    # qwen_sparse_attn_backend.py), so no scale is applied here either.
    out_dtype = out_k.dtype.element_ty
    tl.store(
        out_k + dst,
        tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )
    tl.store(
        out_v + dst,
        tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, without the packed cu_seqlens prefix sum."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    """Gather the selected K/V rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the strided (page-aligned) layout used by the paged
    decode kernel: row ``b`` owns ``[cu_k[b], cu_k[b] + zero_fill_cols)`` and every slot
    past its valid rows is zero-filled. Paged kernels read whole pages and multiply the
    masked probabilities into V, so stale or uninitialized bytes there (NaN/Inf bit
    patterns) would otherwise leak into the output. ``0`` keeps the compact layout for
    the varlen fallback, whose rows are packed back-to-back.

    ``out_k``/``out_v`` may use a wider dtype than the pool (bf16 scratch for an FP8
    pool); rows are converted while gathering.

    Both layouts assume the valid entries of each ``indices`` row are contiguous at
    the front (``expand_qsa_block_indices`` sorts them that way): ``valid_count`` is a
    count, not a mask, so a ``-1`` in the middle of a row would shift the packing.
    """
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    _compact_kv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        num_warps=8,
    )


@triton.jit(
    do_not_specialize=[
        "batch",
        "sr0",
        "topk_limit",
        "spo_b",
        "spo_h",
        "spo_s",
        "spl_b",
        "spl_h",
        "si_m",
    ]
)
def _sparse_gqa_paged_decode_fp8_pairs(
    q,
    k_pairs_ptr,
    v_pairs_ptr,
    out,
    gate,
    partial_out,
    partial_lse,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    qpos_ptr,
    scale,
    batch,
    sr0,
    topk_limit,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sg_m: tl.constexpr,
    sg_h: tl.constexpr,
    sg_d: tl.constexpr,
    spo_b,
    spo_h,
    spo_s,
    spl_b,
    spl_h,
    si_m,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    EXPAND: tl.constexpr,
    RATIO: tl.constexpr,
    IB: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ALIGNED_8: tl.constexpr,
):
    """Direct-read SM90+ FP8-E4M3 paged sparse decode/verify kernel.

    Reads scattered FP8 K/V slots straight from the pool via int16 pairs +
    pack=2 inline PTX cvt.rn.f16x2.e4m3x2, eliminating _qsa_fused_kv_prepare
    copies and optional output_gate epilogue passes. Supports inline block
    expansion (EXPAND=True) and split-K parallelism (NUM_SPLITS > 1).
    """
    row_group = tl.program_id(0)
    split_id = tl.program_id(1)
    group = row_group % NUM_KV_HEADS
    row = row_group // NUM_KV_HEADS
    if row >= batch:
        return

    length = tl.load(seq_lens + row).to(tl.int32)
    if EXPAND:
        n_blocks = tl.minimum(length // RATIO, IB) * RATIO
        qpos = tl.load(qpos_ptr + row).to(tl.int32)
        tail_start = ((qpos + 1) // RATIO) * RATIO
        tail_count = tl.maximum(0, tl.minimum(qpos + 1, length) - tail_start)
        row_cols = tl.minimum(topk_limit, n_blocks + tl.minimum(tail_count, RATIO - 1))
    else:
        n_blocks = 0
        qpos = 0
        tail_start = 0
        # Keep full topk_limit when EXPAND=False because QSAMTPSharedSparseIndices
        # appends drafted tail positions at the end of the row past -1 padding.
        row_cols = topk_limit

    num_tiles = tl.cdiv(row_cols, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, NUM_SPLITS)
    tile_start = split_id * tiles_per_split
    tile_end = tl.minimum(tile_start + tiles_per_split, num_tiles)

    offs_h = tl.arange(0, BLOCK_M)
    head_start = group * GROUP_SIZE
    head_mask_1d = offs_h < GROUP_SIZE

    if tile_start >= tile_end:
        if NUM_SPLITS > 1:
            tl.store(
                partial_lse
                + row.to(tl.int64) * spl_b
                + (head_start + offs_h).to(tl.int64) * spl_h
                + split_id,
                tl.full([BLOCK_M], -1.0e20, tl.float32),
                mask=head_mask_1d,
            )
        else:
            offs_d = tl.arange(0, HEAD_DIM)
            tl.store(
                out
                + row.to(tl.int64) * so_m
                + (head_start + offs_h[:, None]).to(tl.int64) * so_h
                + offs_d[None, :] * so_d,
                tl.zeros([BLOCK_M, HEAD_DIM], dtype=out.dtype.element_ty),
                mask=head_mask_1d[:, None],
            )
        return

    offs_half_d = tl.arange(0, HALF_DIM)
    head_mask = head_mask_1d[:, None]
    q_pair_base = (
        q
        + row.to(tl.int64) * sq_m
        + (head_start + offs_h[:, None]).to(tl.int64) * sq_h
        + (2 * offs_half_d[None, :]) * sq_d
    )
    q_even = tl.load(q_pair_base, mask=head_mask, other=0.0)
    q_odd = tl.load(q_pair_base + sq_d, mask=head_mask, other=0.0)
    q_dtype = q_even.dtype
    score_scale = scale * 1.4426950408889634

    req = tl.load(req_indices + row).to(tl.int64)
    r2t_row = req_to_token + req * sr0
    k_base = k_pairs_ptr + group * sk_h
    v_base = v_pairs_ptr + group * sv_h
    idx_row = indices + row.to(tl.int64) * si_m

    max_value = tl.full([BLOCK_M], -1.0e20, tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    acc_even = tl.zeros([BLOCK_M, HALF_DIM], tl.float32)
    acc_odd = tl.zeros([BLOCK_M, HALF_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    for tile in tl.range(tile_start, tile_end, num_stages=NUM_STAGES):
        c = tile * BLOCK_N + offs_n
        if EXPAND:
            blocks = tl.load(idx_row + (c // RATIO), mask=c < IB * RATIO, other=-1)
            expanded = blocks * RATIO + (c % RATIO)
            tail_off = c - n_blocks
            tail = tail_start + tail_off
            pos = tl.where(
                (c < n_blocks) & (blocks >= 0),
                expanded,
                tl.where(
                    (tail_off >= 0) & (tail_off < RATIO - 1) & (tail < qpos + 1),
                    tail,
                    -1,
                ),
            )
        else:
            pos = tl.load(idx_row + c, mask=c < topk_limit, other=-1)
        valid = (pos >= 0) & (pos < length) & (c < topk_limit)
        safe_pos = tl.maximum(pos, 0).to(tl.int64)
        slot = tl.load(r2t_row + safe_pos, mask=valid, other=0).to(tl.int64)
        safe_slot = tl.maximum(slot, 0)

        k_tok_off = safe_slot * sk_n
        v_tok_off = safe_slot * sv_n
        if ALIGNED_8:
            k_tok_off = tl.multiple_of(k_tok_off, 8)
            v_tok_off = tl.multiple_of(v_tok_off, 8)
        key_pairs = tl.trans(
            tl.load(
                k_base + k_tok_off[:, None] + offs_half_d[None, :],
                mask=valid[:, None],
                other=0,
            )
        )
        value_pairs = tl.load(
            v_base + v_tok_off[:, None] + offs_half_d[None, :],
            mask=valid[:, None],
            other=0,
        )

        scores = tl.dot(
            q_even, _e4m3_pairs_to_fp16(key_pairs, False).to(q_dtype)
        ) + tl.dot(q_odd, _e4m3_pairs_to_fp16(key_pairs, True).to(q_dtype))
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)

        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        probs_fp16 = probabilities.to(tl.float16)

        acc_even = tl.dot(
            probs_fp16,
            _e4m3_pairs_to_fp16(value_pairs, False),
            acc=acc_even * alpha[:, None],
        )
        acc_odd = tl.dot(
            probs_fp16,
            _e4m3_pairs_to_fp16(value_pairs, True),
            acc=acc_odd * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    inv_norm = 1.0 / tl.maximum(normalizer, 1.0e-20)
    out_even = tl.where(has_values[:, None], acc_even * inv_norm[:, None], 0.0)
    out_odd = tl.where(has_values[:, None], acc_odd * inv_norm[:, None], 0.0)
    output = tl.interleave(out_even, out_odd)

    if NUM_SPLITS > 1:
        lse = tl.where(
            has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -1.0e20
        )
        tl.store(
            partial_lse
            + row.to(tl.int64) * spl_b
            + (head_start + offs_h).to(tl.int64) * spl_h
            + split_id,
            lse,
            mask=head_mask_1d,
        )
        offs_d = tl.arange(0, HEAD_DIM)
        tl.store(
            partial_out
            + row.to(tl.int64) * spo_b
            + (head_start + offs_h[:, None]).to(tl.int64) * spo_h
            + split_id.to(tl.int64) * spo_s
            + offs_d[None, :],
            output,
            mask=head_mask,
        )
    else:
        offs_d = tl.arange(0, HEAD_DIM)
        if HAS_GATE:
            g_vals = tl.load(
                gate
                + row.to(tl.int64) * sg_m
                + (head_start + offs_h[:, None]).to(tl.int64) * sg_h
                + offs_d[None, :] * sg_d,
                mask=head_mask,
                other=0.0,
            ).to(tl.float32)
            output = output * tl.sigmoid(g_vals)
        tl.store(
            out
            + row.to(tl.int64) * so_m
            + (head_start + offs_h[:, None]).to(tl.int64) * so_h
            + offs_d[None, :] * so_d,
            output.to(out.dtype.element_ty),
            mask=head_mask,
        )


@triton.jit(
    do_not_specialize=["batch", "spo_b", "spo_h", "spo_s", "spl_b", "spl_h"]
)
def _sparse_gqa_paged_decode_reduce(
    partial_out,
    partial_lse,
    out,
    gate,
    batch,
    spo_b,
    spo_h,
    spo_s,
    spl_b,
    spl_h,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sg_m: tl.constexpr,
    sg_h: tl.constexpr,
    sg_d: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % NUM_Q_HEADS
    row = pid // NUM_Q_HEADS
    if row >= batch:
        return

    offs_s = tl.arange(0, NUM_SPLITS)
    lse_base = partial_lse + row.to(tl.int64) * spl_b + head.to(tl.int64) * spl_h
    lses = tl.load(lse_base + offs_s)
    valid = lses > -1.0e19
    max_lse = tl.max(tl.where(valid, lses, -1.0e20), axis=0)
    weights = tl.where(valid, tl.math.exp2(lses - max_lse), 0.0)
    sum_w = tl.sum(weights, axis=0)

    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    po_base = partial_out + row.to(tl.int64) * spo_b + head.to(tl.int64) * spo_h
    for s in tl.static_range(NUM_SPLITS):
        lse_s = tl.load(lse_base + s)
        if lse_s > -1.0e19:
            w_s = tl.math.exp2(lse_s - max_lse)
            p_s = tl.load(po_base + s * spo_s + offs_d)
            acc += w_s * p_s

    output = tl.where(sum_w > 0.0, acc / tl.maximum(sum_w, 1.0e-20), 0.0)
    if HAS_GATE:
        g_vals = tl.load(
            gate
            + row.to(tl.int64) * sg_m
            + head.to(tl.int64) * sg_h
            + offs_d * sg_d
        ).to(tl.float32)
        output = output * tl.sigmoid(g_vals)
    tl.store(
        out + row.to(tl.int64) * so_m + head.to(tl.int64) * so_h + offs_d * so_d,
        output.to(out.dtype.element_ty),
    )


def _choose_fp8_decode_splits(batch: int, num_kv_heads: int, device: torch.device) -> int:
    del device
    base = max(batch * num_kv_heads, 1)
    # B300-measured split-K crossover (BLOCK_N=64, num_warps=4, num_stages=5):
    # - M <= 8 and 32 < M <= 128 (incl. c=64 draft): splits=1 avoids reduce
    #   launch overhead (42.4-43.2 us vs 53.4-59.1 us).
    # - 8 < M <= 32: splits=2 fills SM waves (53.3 us vs 68.6 us).
    # - M > 128 (incl. c=64 verify M=256): splits=4 hides gather latency (64.4 us).
    if base <= 8 or (32 < base <= 128):
        return 1
    if base <= 32:
        return 2
    return 4


def can_use_sparse_gqa_paged_decode_fp8(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    req_to_token: torch.Tensor,
) -> bool:
    ok, packed, _ = _can_use_fp8_pairs(q, k_pool, v_pool)
    return ok and packed and req_to_token.is_cuda and req_to_token.stride(-1) == 1


def sparse_gqa_paged_decode_fp8_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    req_to_token: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    *,
    compress_ratio: int = 1,
    query_positions: Optional[torch.Tensor] = None,
    output_gate: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    num_splits: Optional[int] = None,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 5,
) -> torch.Tensor:
    """Direct-read FP8-E4M3 paged sparse decode/verify attention with split-K
    and optional fused output_gate sigmoid."""
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    if out is None:
        out = torch.empty_like(q)
    if batch == 0:
        return out

    expand = compress_ratio > 1
    if expand:
        if query_positions is None:
            raise ValueError("query_positions required when compress_ratio > 1")
        ib = topk_indices.shape[1]
        topk_limit = ib * compress_ratio + compress_ratio - 1
        qpos_arg = query_positions
    else:
        ib = 0
        topk_limit = topk_indices.shape[1]
        qpos_arg = seq_lens

    _, _, aligned_8 = _can_use_fp8_pairs(q, k_pool, v_pool)
    k_arg = k_pool.view(torch.int16)
    v_arg = v_pool.view(torch.int16)
    splits = (
        int(num_splits)
        if num_splits is not None
        else _choose_fp8_decode_splits(batch, num_kv_heads, q.device)
    )
    gate_3d = _normalize_output_gate(output_gate, batch, num_q_heads, head_dim)
    has_gate = gate_3d is not None
    gate_arg = gate_3d if has_gate else q

    if splits > 1:
        if (
            workspace is not None
            and workspace[0].shape[0] >= batch
            and workspace[0].shape[1] == num_q_heads
            and workspace[0].shape[2] >= splits
            and workspace[0].shape[3] == head_dim
            and workspace[1].shape[0] >= batch
            and workspace[1].shape[1] == num_q_heads
            and workspace[1].shape[2] >= splits
        ):
            p_out = workspace[0][:batch, :, :splits, :]
            p_lse = workspace[1][:batch, :, :splits]
        else:
            p_out = torch.empty(
                (batch, num_q_heads, splits, head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            p_lse = torch.empty(
                (batch, num_q_heads, splits), dtype=torch.float32, device=q.device
            )
        spo_b, spo_h, spo_s = p_out.stride(0), p_out.stride(1), p_out.stride(2)
        spl_b, spl_h = p_lse.stride(0), p_lse.stride(1)
    else:
        p_out = out
        p_lse = seq_lens
        spo_b = spo_h = spo_s = spl_b = spl_h = 0

    _sparse_gqa_paged_decode_fp8_pairs[(batch * num_kv_heads, splits)](
        q,
        k_arg,
        v_arg,
        out,
        gate_arg,
        p_out,
        p_lse,
        req_to_token,
        req_indices,
        topk_indices,
        seq_lens,
        qpos_arg,
        scale,
        batch,
        req_to_token.stride(0),
        topk_limit,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_arg.stride(0),
        k_arg.stride(1),
        v_arg.stride(0),
        v_arg.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        gate_arg.stride(0) if has_gate else 0,
        gate_arg.stride(1) if has_gate else 0,
        gate_arg.stride(2) if has_gate else 0,
        spo_b,
        spo_h,
        spo_s,
        spl_b,
        spl_h,
        topk_indices.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        NUM_STAGES=num_stages,
        NUM_SPLITS=splits,
        EXPAND=expand,
        RATIO=compress_ratio if expand else 1,
        IB=ib,
        HAS_GATE=has_gate,
        ALIGNED_8=aligned_8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if splits > 1:
        _sparse_gqa_paged_decode_reduce[(batch * num_q_heads,)](
            p_out,
            p_lse,
            out,
            gate_arg,
            batch,
            spo_b,
            spo_h,
            spo_s,
            spl_b,
            spl_h,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            gate_arg.stride(0) if has_gate else 0,
            gate_arg.stride(1) if has_gate else 0,
            gate_arg.stride(2) if has_gate else 0,
            NUM_Q_HEADS=num_q_heads,
            HEAD_DIM=head_dim,
            NUM_SPLITS=splits,
            HAS_GATE=has_gate,
            num_warps=4,
        )
    return out


__all__ = [
    "can_use_sparse_gqa_paged_decode_fp8",
    "pack_qsa_prefill_kv",
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_gqa_packed_decode_triton",
    "sparse_gqa_paged_decode_fp8_triton",
    "sparse_gqa_paged_prefill_triton",
]
