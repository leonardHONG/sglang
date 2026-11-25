"""CUDA kernels and tensor transforms for simple QSA."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils import is_hip, is_npu

_is_npu = is_npu()


def average_pool_qsa_keys(key_groups: torch.Tensor) -> torch.Tensor:
    """FP32-average complete key groups shaped ``[groups, ratio, kv_heads, dim]``."""

    if key_groups.ndim != 4:
        raise ValueError(
            "QSA key groups must be [groups, ratio, kv_heads, head_dim], "
            f"got {key_groups.shape}"
        )
    return key_groups.float().mean(dim=1).to(key_groups.dtype)


def qsa_fast_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select compressed blocks with platform kernels or a Torch reference.

    Return int32 indices of shape [rows, topk], relative to each row's start.
    Valid indices precede any -1 padding.
    """

    lengths = (row_ends - row_starts).to(device=logits.device, dtype=torch.int32)
    starts = row_starts.to(device=logits.device, dtype=torch.int32)
    if _is_npu:
        from sgl_kernel_npu.operators.indexer import qsa_topk as fast_topk

        # Upstream supplies contiguous bounds and finite scores within each
        # valid interval. Unsupported metadata/errors must not silently fall back.
        return fast_topk(logits, lengths, topk, starts)

    if logits.is_cuda:
        if is_batch_invariant_mode_enabled() and not is_hip():
            from flashinfer import TopKTieBreak, top_k_ragged_transform

            return top_k_ragged_transform(
                logits,
                torch.zeros_like(starts),
                lengths,
                topk,
                deterministic=True,
                tie_break=TopKTieBreak.SMALL,
                dsa_graph_safe=True,
                row_starts=starts,
            )
        if topk == 512:
            # Prefer the JIT kernel: it ships with the sglang python package,
            # so top-k 512 works regardless of the installed sgl_kernel version.
            from sglang.kernels.ops.attention.fast_topk import fast_topk

            return fast_topk(logits, lengths, topk=512, row_starts=starts)

        from sgl_kernel import top_k as top_k_module

        supported_topk = getattr(top_k_module, "_FAST_TOPK_SUPPORTED_K", (2048,))
        if topk in supported_topk:
            return top_k_module.fast_topk_v2(
                logits, lengths, topk=topk, row_starts=starts
            )
        raise ValueError(
            f"QSA top-k {topk} is unsupported by sgl_kernel; "
            f"supported values are {supported_topk}"
        )

    # CPU/reference path mirrors the CUDA operator's fixed-width, relative output.
    output = torch.full(
        (logits.shape[0], topk),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(lengths[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def torch_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed block indices into fixed-width logical token indices."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat([expanded, tail], dim=1)
    # Keep all valid entries contiguous. This is required by the FA2 packing path.
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


@triton.jit
def _expand_qsa_block_indices_kernel(
    block_indices,
    query_positions,
    sequence_lengths,
    output,
    block_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    FINAL_TOPK: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, OUTPUT_BLOCK_SIZE)
    sequence_length = tl.load(sequence_lengths + row)

    source_block_cols = cols // COMPRESS_RATIO
    offsets = cols % COMPRESS_RATIO
    blocks = tl.load(
        block_indices + row * block_stride + source_block_cols,
        mask=(cols < TOKEN_TOPK) & (source_block_cols < BLOCK_TOPK),
        other=-1,
    )
    expanded = blocks * COMPRESS_RATIO + offsets
    expanded_valid = (
        (cols < TOKEN_TOPK)
        & (blocks >= 0)
        & (expanded >= 0)
        & (expanded < sequence_length)
    )

    valid_block_count = tl.sum(
        (
            (cols < BLOCK_TOPK)
            & (
                tl.load(
                    block_indices + row * block_stride + cols,
                    mask=cols < BLOCK_TOPK,
                    other=-1,
                )
                >= 0
            )
        ).to(tl.int32),
        axis=0,
    )
    valid_token_count = tl.minimum(valid_block_count * COMPRESS_RATIO, TOKEN_TOPK)

    query_position = tl.load(query_positions + row)
    visible_tokens = query_position + 1
    tail_start = (visible_tokens // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_offset = cols - valid_token_count
    tail_count = visible_tokens - tail_start
    tail = tail_start + tail_offset
    tail_valid = (
        (tail_offset >= 0)
        & (tail_offset < COMPRESS_RATIO - 1)
        & (tail_offset < tail_count)
        & (tail < sequence_length)
    )

    result = tl.where(
        expanded_valid & (cols < valid_token_count),
        expanded,
        tl.where(tail_valid, tail, -1),
    )
    tl.store(
        output + row * output_stride + cols,
        result,
        mask=cols < FINAL_TOPK,
    )


def triton_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA fast path for fast_topk_v2 output (valid blocks precede -1 padding)."""
    rows, block_topk = block_indices.shape
    final_topk = token_topk + compress_ratio - 1
    if out is not None:
        if (
            out.shape != (rows, final_topk)
            or out.dtype != torch.int32
            or out.device != block_indices.device
            or out.stride(1) != 1
        ):
            raise ValueError(
                f"invalid expand_qsa_block_indices out buffer: "
                f"shape={tuple(out.shape)} dtype={out.dtype} stride={out.stride()}"
            )
        output = out
    else:
        output = torch.empty(
            (rows, final_topk), dtype=torch.int32, device=block_indices.device
        )
    if rows == 0:
        return output
    _expand_qsa_block_indices_kernel[(rows,)](
        block_indices,
        query_positions,
        sequence_lengths,
        output,
        block_indices.stride(0),
        output.stride(0),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        FINAL_TOPK=final_topk,
        OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
        num_warps=8,
    )
    return output


@triton.jit(do_not_specialize=["table_stride", "total_blocks"])
def _gather_qsa_compressed_keys_kernel(
    compressed,
    output,
    token_slot_table,
    cu_blocks,
    table_stride,
    total_blocks,
    RATIO: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    sequence = tl.program_id(1)
    start = tl.load(cu_blocks + sequence)
    count = tl.load(cu_blocks + sequence + 1) - start
    blocks = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    # The device offsets must agree with the host sizes; never write past them.
    valid = (blocks < count) & (start + blocks < total_blocks)
    # A block's compressed slot is its first token's slot // ratio: the
    # allocator is page-aligned, so the group is contiguous in one page.
    first = tl.load(
        token_slot_table + sequence.to(tl.int64) * table_stride + blocks * RATIO,
        mask=valid,
        other=0,
    ).to(tl.int64)
    cols = tl.arange(0, ROW)
    keys = tl.load(
        compressed + (first // RATIO)[:, None] * ROW + cols[None, :],
        mask=valid[:, None],
    )
    rows = (start + blocks).to(tl.int64)
    tl.store(output + rows[:, None] * ROW + cols[None, :], keys, mask=valid[:, None])


def gather_qsa_compressed_keys(
    compressed: torch.Tensor,
    token_slot_table: torch.Tensor,
    cu_blocks: torch.Tensor,
    total_blocks: int,
    max_blocks: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Every sequence's complete compressed keys, packed in sequence order, in
    one launch: rows ``cu_blocks[i]:cu_blocks[i + 1]`` are sequence ``i``'s
    blocks. ``total_blocks``/``max_blocks`` are host ints."""
    heads, dim = compressed.shape[1:]
    output = compressed.new_empty((total_blocks, heads, dim))
    if total_blocks == 0:
        return output
    # Copy raw bytes: one kernel for every cache dtype (bf16 or FP8).
    row = heads * dim * compressed.element_size()
    assert compressed.is_contiguous() and row == triton.next_power_of_2(row)
    assert token_slot_table.stride(-1) == 1
    block_b = 64
    _gather_qsa_compressed_keys_kernel[
        (triton.cdiv(max_blocks, block_b), cu_blocks.shape[0] - 1)
    ](
        compressed.view(torch.uint8),
        output.view(torch.uint8),
        token_slot_table,
        cu_blocks,
        token_slot_table.stride(0),
        total_blocks,
        RATIO=compress_ratio,
        ROW=row,
        BLOCK_B=block_b,
        num_warps=4,
    )
    return output


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand compressed blocks with platform kernels or the Torch fallback."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")
    if _is_npu:
        from sgl_kernel_npu.operators.indexer import expand_qsa_blocks as expand_blocks

        # The package wrapper owns the model contract and its internal dispatch.
        # Unsupported metadata must not silently enter the generic reference.
        result = expand_blocks(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio,
            token_topk,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result
    if block_indices.is_cuda:
        # The Triton kernel loads positions/lengths as scalars, so any integer
        # dtype works; skip the int64 conversion copies.
        return triton_expand_qsa_block_indices(
            block_indices.contiguous(),
            query_positions.to(device=block_indices.device).contiguous(),
            sequence_lengths.to(device=block_indices.device).contiguous(),
            compress_ratio,
            token_topk,
            out=out,
        )
    result = torch_expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio,
        token_topk,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for sparse GQA over physical token slots."""

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if token_slots.ndim != 2 or token_slots.shape[0] != q.shape[0]:
        raise ValueError(
            "token slots must be [query_tokens, selected_tokens], got "
            f"{token_slots.shape}"
        )
    if q.shape[-1] != k_cache.shape[-1] or q.shape[-1] != v_cache.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    return qsa_sparse_attention_reference(
        q, k_cache, v_cache, token_slots, softmax_scale
    )


def qsa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Device-agnostic sparse GQA reference."""

    scale = softmax_scale or q.shape[-1] ** -0.5
    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        valid = token_slots[row] >= 0
        slots = token_slots[row, valid].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        values = v_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype)
        )
    return torch.stack(outputs)


@triton.jit(do_not_specialize=["num_tokens", "batch_size"])
def _qsa_prefill_token_meta_kernel(
    extend_cu_ends,
    seq_lens,
    req_pool_indices,
    compressed_cu_seqlens,
    logical_positions,
    out_token_to_batch_idx,
    out_row_starts,
    out_row_ends,
    out_row_lengths,
    out_row_seq_lens,
    out_pending_ring_slots,
    num_tokens,
    batch_size,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    # Binary search smallest b in [0, batch_size - 1] with offs < extend_cu_ends[b].
    lo = tl.zeros((BLOCK,), dtype=tl.int32)
    hi = tl.full((BLOCK,), batch_size - 1, dtype=tl.int32)
    steps = batch_size
    while steps > 1:
        mid = (lo + hi) // 2
        bound = tl.load(extend_cu_ends + mid, mask=mask, other=0)
        go_right = offs >= bound
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
        steps = (steps + 1) // 2
    b = lo

    seq_len = tl.load(seq_lens + b, mask=mask, other=0).to(tl.int32)
    req_slot = tl.load(req_pool_indices + b, mask=mask, other=0).to(tl.int64)
    row_start = tl.load(compressed_cu_seqlens + b, mask=mask, other=0).to(tl.int32)
    next_start = tl.load(compressed_cu_seqlens + b + 1, mask=mask, other=0).to(tl.int32)
    max_blocks = next_start - row_start
    pos = tl.load(logical_positions + offs, mask=mask, other=0).to(tl.int64)

    pos_i32 = pos.to(tl.int32)
    visible_blocks = (pos_i32 + 1) // RATIO
    row_len = tl.minimum(visible_blocks, max_blocks)
    row_end = row_start + row_len

    tail_base = ((seq_len // RATIO) * RATIO).to(tl.int64)
    rem = pos % RATIO
    ring_slot = tl.where(pos >= tail_base, req_slot * RATIO + rem, rem)

    tl.store(out_token_to_batch_idx + offs, b, mask=mask)
    tl.store(out_row_starts + offs, row_start, mask=mask)
    tl.store(out_row_ends + offs, row_end, mask=mask)
    tl.store(out_row_lengths + offs, row_len, mask=mask)
    tl.store(out_row_seq_lens + offs, seq_len, mask=mask)
    tl.store(out_pending_ring_slots + offs, ring_slot, mask=mask)


@triton.jit(do_not_specialize=["total_groups", "batch_size", "table_stride", "num_tokens"])
def _qsa_prefill_write_plan_kernel(
    group_cu_ends,
    start_blocks,
    row_token_starts,
    token_slot_table,
    out_write_locs,
    out_group_end_positions,
    out_group_sequence_ids,
    out_group_member_rows,
    out_compress_group_locs,
    total_groups,
    batch_size,
    table_stride,
    num_tokens,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_groups
    lo = tl.zeros((BLOCK,), dtype=tl.int32)
    hi = tl.full((BLOCK,), batch_size - 1, dtype=tl.int32)
    steps = batch_size
    while steps > 1:
        mid = (lo + hi) // 2
        bound = tl.load(group_cu_ends + mid, mask=mask, other=0)
        go_right = offs >= bound
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
        steps = (steps + 1) // 2
    b = lo

    prev_end = tl.where(
        b > 0,
        tl.load(group_cu_ends + tl.maximum(b - 1, 0), mask=mask & (b > 0), other=0),
        0,
    )
    local_g = offs - prev_end
    start_b = tl.load(start_blocks + b, mask=mask, other=0)
    tok_start = tl.load(row_token_starts + b, mask=mask, other=0)
    block_idx = start_b + local_g

    slot0 = tl.load(
        token_slot_table
        + b.to(tl.int64) * table_stride
        + block_idx.to(tl.int64) * RATIO,
        mask=mask,
        other=0,
    ).to(tl.int64)
    write_loc = (slot0 // RATIO).to(tl.int32)
    group_end_pos = (block_idx.to(tl.int64) * RATIO + (RATIO - 1))
    member_row = (tok_start + local_g * RATIO).to(tl.int64)

    tl.store(out_write_locs + offs, write_loc, mask=mask)
    tl.store(out_group_end_positions + offs, group_end_pos, mask=mask)
    tl.store(out_group_sequence_ids + offs, b.to(tl.int64), mask=mask)
    tl.store(out_group_member_rows + offs, member_row, mask=mask)

    max_tok = num_tokens - 1
    base_i32 = member_row.to(tl.int32)
    offs64 = offs.to(tl.int64)
    for r in tl.static_range(RATIO):
        tl.store(
            out_compress_group_locs + offs64 * RATIO + r,
            tl.minimum(base_i32 + r, max_tok),
            mask=mask,
        )


def build_qsa_prefill_metadata_fused(
    *,
    extend_lens_cpu: list[int],
    seq_lens_cpu: list[int],
    sequence_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    token_slot_table: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
):
    """Build all per-token and per-group QSA prefill metadata tensors in 2 Triton
    launches + 1 packed pinned HtoD instead of ~85 tiny PyTorch ops."""
    device = sequence_lengths.device
    batch_size = len(extend_lens_cpu)
    num_tokens = 0
    total_groups = 0
    comp_total = 0

    # Pack 4 small int32 host arrays into one pinned transfer:
    # [extend_cu_seqlens(B+1), compressed_cu_seqlens(B+1),
    #  start_blocks(B), group_cu_ends(B)]
    # row_token_starts = extend_cu_seqlens[:-1], extend_cu_ends = extend_cu_seqlens[1:].
    packed_len = 4 * batch_size + 2
    packed_cpu = torch.empty(
        packed_len, dtype=torch.int32, pin_memory=device.type == "cuda"
    )
    arr = packed_cpu.numpy()
    off_ext = 0
    off_comp = batch_size + 1
    off_sb = 2 * batch_size + 2
    off_grp = 3 * batch_size + 2

    arr[off_ext] = 0
    arr[off_comp] = 0
    for i in range(batch_size):
        ext_l = int(extend_lens_cpu[i])
        seq_l = int(seq_lens_cpu[i])
        pref_l = max(0, seq_l - ext_l)
        sb = pref_l // compress_ratio
        eb = seq_l // compress_ratio
        cnt = max(0, eb - sb)
        num_tokens += ext_l
        arr[off_ext + i + 1] = num_tokens
        comp_total += eb
        arr[off_comp + i + 1] = comp_total
        arr[off_sb + i] = sb
        total_groups += cnt
        arr[off_grp + i] = total_groups

    packed_dev = packed_cpu.to(device, non_blocking=True)
    extend_cu_seqlens = packed_dev[off_ext : off_ext + batch_size + 1]
    extend_cu_ends = extend_cu_seqlens[1:]
    row_token_starts = extend_cu_seqlens[:-1]
    compressed_cu_seqlens = packed_dev[off_comp : off_comp + batch_size + 1]
    start_blocks = packed_dev[off_sb : off_sb + batch_size]
    group_cu_ends = packed_dev[off_grp : off_grp + batch_size]

    token_to_batch_idx = torch.empty(num_tokens, dtype=torch.int32, device=device)
    prefill_row_starts = torch.empty(num_tokens, dtype=torch.int32, device=device)
    prefill_row_ends = torch.empty(num_tokens, dtype=torch.int32, device=device)
    prefill_row_lengths = torch.empty(num_tokens, dtype=torch.int32, device=device)
    prefill_row_sequence_lengths = torch.empty(
        num_tokens, dtype=torch.int32, device=device
    )
    pending_ring_slots = torch.empty(num_tokens, dtype=torch.int64, device=device)

    block = 256
    _qsa_prefill_token_meta_kernel[(triton.cdiv(num_tokens, block),)](
        extend_cu_ends,
        sequence_lengths,
        req_pool_indices,
        compressed_cu_seqlens,
        logical_positions,
        token_to_batch_idx,
        prefill_row_starts,
        prefill_row_ends,
        prefill_row_lengths,
        prefill_row_sequence_lengths,
        pending_ring_slots,
        num_tokens,
        batch_size,
        RATIO=compress_ratio,
        BLOCK=block,
        num_warps=4,
    )

    if total_groups > 0:
        write_locs = torch.empty(total_groups, dtype=torch.int32, device=device)
        group_end_positions = torch.empty(
            total_groups, dtype=torch.int64, device=device
        )
        group_sequence_ids = torch.empty(total_groups, dtype=torch.int64, device=device)
        group_member_rows = torch.empty(total_groups, dtype=torch.int64, device=device)
        prefill_compress_group_locs = torch.empty(
            (total_groups, compress_ratio), dtype=torch.int32, device=device
        )
        _qsa_prefill_write_plan_kernel[(triton.cdiv(total_groups, block),)](
            group_cu_ends,
            start_blocks,
            row_token_starts,
            token_slot_table,
            write_locs,
            group_end_positions,
            group_sequence_ids,
            group_member_rows,
            prefill_compress_group_locs,
            total_groups,
            batch_size,
            token_slot_table.stride(0),
            num_tokens,
            RATIO=compress_ratio,
            BLOCK=block,
            num_warps=4,
        )
    else:
        write_locs = torch.empty(0, dtype=torch.int32, device=device)
        group_end_positions = torch.empty(0, dtype=torch.int64, device=device)
        group_sequence_ids = torch.empty(0, dtype=torch.int64, device=device)
        group_member_rows = torch.empty(0, dtype=torch.int64, device=device)
        prefill_compress_group_locs = None

    return (
        token_to_batch_idx,
        write_locs,
        group_end_positions,
        group_sequence_ids,
        group_member_rows,
        pending_ring_slots,
        prefill_row_starts,
        prefill_row_ends,
        compressed_cu_seqlens,
        prefill_row_sequence_lengths,
        prefill_row_lengths,
        prefill_compress_group_locs,
        extend_cu_seqlens,
    )


__all__ = [
    "average_pool_qsa_keys",
    "build_qsa_prefill_metadata_fused",
    "expand_qsa_block_indices",
    "gather_qsa_compressed_keys",
    "torch_expand_qsa_block_indices",
    "triton_expand_qsa_block_indices",
    "qsa_fast_topk",
    "qsa_sparse_attention",
    "qsa_sparse_attention_reference",
]
