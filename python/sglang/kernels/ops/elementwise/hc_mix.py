from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# Empirically measured crossover on B300 (H=2560, HC=4, LR=320):
# - M <= 512 (decode / spec-verify): single-kernel fused up+gate+mix beats
#   cuBLAS + hc_gate_mix by 1.25x-1.36x (and torch.compile by 1.46x-1.59x).
# - M > 512 (prefill): cuBLAS 2-CTA tcgen05 up-GEMM + Triton hc_gate_mix
#   achieves ~0.15 ms/call at M=8244 (2.3x+ faster than Wave 1 unpadded).
_FUSED_UP_MAX_TOKENS = 512


@triton.jit(do_not_specialize=["num_tokens"])
def _hc_silu_kernel(
    x_ptr,
    y_ptr,
    num_tokens,
    stride_x,
    stride_y: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_tokens:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < RANK
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32) * (
        1.0 / HC
    )
    y = x * tl.sigmoid(x)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


@triton.jit(do_not_specialize=["num_tokens"])
def _hc_gate_mix_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    num_tokens,
    stride_x,
    stride_g,
    stride_y,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """Fused sigmoid-gate + 4-stream mean reduction with PDL and streaming loads."""
    row = tl.program_id(0).to(tl.int64)
    if row >= num_tokens:
        return
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_inner < HIDDEN

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HIDDEN + offs_inner
        g = tl.load(g_ptr + row * stride_g + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc *= 1.0 / HC

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs_inner, acc, mask=mask)


@triton.jit(do_not_specialize=["num_tokens"])
def _hc_mix_up_gate_kernel(
    lowrank_ptr,
    weight_ptr,
    xn_ptr,
    output_ptr,
    num_tokens,
    stride_lowrank_m,
    stride_lowrank_k: tl.constexpr,
    stride_weight_n: tl.constexpr,
    stride_weight_k: tl.constexpr,
    stride_xn_m,
    stride_xn_n: tl.constexpr,
    HIDDEN: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """Single-kernel fused HyperConnection silu + up-projection + sigmoid gate + mix.

    Optimal for decode / spec-verify token counts (M <= 512) where avoiding the
    [M, HC * HIDDEN] gate round-trip and extra kernel launches dominates.
    """
    GROUP_M: tl.constexpr = 8
    FEATS_PER_TILE: tl.constexpr = BLOCK_N // HC
    num_m = tl.cdiv(num_tokens, BLOCK_M)
    num_h: tl.constexpr = HIDDEN // FEATS_PER_TILE

    tile = tl.program_id(0)
    first_m = (tile // (GROUP_M * num_h)) * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    in_group = tile % (GROUP_M * num_h)
    tile_m = first_m + in_group % group_m
    tile_h = in_group // group_m

    rows = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    row_mask = rows < num_tokens
    cols = tl.arange(0, BLOCK_N)
    features = tile_h * FEATS_PER_TILE + (cols % FEATS_PER_TILE)
    gate_cols = (cols // FEATS_PER_TILE) * HIDDEN + features
    k_offs = tl.arange(0, BLOCK_K)

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(RANK // BLOCK_K):
        offset = block * BLOCK_K + k_offs
        a = tl.load(
            lowrank_ptr
            + rows[:, None] * stride_lowrank_m
            + offset[None, :] * stride_lowrank_k,
            mask=row_mask[:, None],
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + gate_cols[:, None] * stride_weight_n
            + offset[None, :] * stride_weight_k,
        )
        if FUSE_SILU:
            a_f32 = a.to(tl.float32) * (1.0 / HC)
            a = (a_f32 * tl.sigmoid(a_f32)).to(w.dtype)
        acc = tl.dot(a, tl.trans(w), acc)

    gate = tl.sigmoid(acc.to(output_ptr.dtype.element_ty).to(tl.float32))
    x = tl.load(
        xn_ptr + rows[:, None] * stride_xn_m + gate_cols[None, :] * stride_xn_n,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    gate = tl.permute(tl.reshape(gate, (BLOCK_M, HC, FEATS_PER_TILE)), (0, 2, 1))
    x = tl.permute(tl.reshape(x, (BLOCK_M, HC, FEATS_PER_TILE)), (0, 2, 1))
    g_even, g_odd = tl.split(tl.reshape(gate, (BLOCK_M, FEATS_PER_TILE, 2, 2)))
    x_even, x_odd = tl.split(tl.reshape(x, (BLOCK_M, FEATS_PER_TILE, 2, 2)))
    g0, g2 = tl.split(g_even)
    g1, g3 = tl.split(g_odd)
    x0, x2 = tl.split(x_even)
    x1, x3 = tl.split(x_odd)
    mixed = tl.fma(g3, x3, tl.fma(g2, x2, tl.fma(g1, x1, g0 * x0))) * (1.0 / HC)

    out_cols = tile_h * FEATS_PER_TILE + tl.arange(0, FEATS_PER_TILE)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output_ptr + rows[:, None] * HIDDEN + out_cols[None, :],
        mixed.to(output_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


def hc_mix_up_gate(
    x: torch.Tensor,
    down_proj: torch.Tensor,
    w_up: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    out: Optional[torch.Tensor] = None,
    *,
    fuse_silu: Optional[bool] = None,
    block_m: int = 64,
    block_n: int = 128,
    block_k: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Fused HyperConnection silu + up-projection + sigmoid gate + stream mix.

    Computes::

        t = silu(down_proj[:, :lowrank] / hc_count)
        gate = sigmoid(t @ w_up.T).unflatten(-1, (hc_count, hidden_size))
        out = (gate * x.unflatten(-1, (hc_count, hidden_size))).mean(dim=-2)

    ``down_proj`` may have extra columns beyond ``lowrank`` (e.g., fused
    ``[down_proj; gate_logits; pad]`` aligned to a multiple of 16).
    """
    if hc_count != 4:
        raise RuntimeError(f"Unsupported hc_count {hc_count}. Supported: 4")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {x.dtype}. Supported: bfloat16, float16")
    feats_per_tile = block_n // hc_count
    if hidden_size <= 0 or hidden_size % feats_per_tile != 0:
        raise RuntimeError(
            f"Unsupported hidden_size {hidden_size}. Must be a multiple of {feats_per_tile}."
        )
    if block_k is None:
        block_k = 32 if lowrank % 32 == 0 else 16
    if lowrank <= 0 or lowrank % block_k != 0:
        raise RuntimeError(
            f"Unsupported lowrank {lowrank}. Must be a multiple of {block_k}."
        )

    x_2d = x.reshape(-1, hc_count * hidden_size)
    down_2d = down_proj.reshape(-1, down_proj.shape[-1])
    num_tokens = x_2d.shape[0]
    if out is None:
        out_2d = x_2d.new_empty((num_tokens, hidden_size))
    else:
        out_2d = out.reshape(-1, hidden_size)
    if num_tokens == 0:
        return out_2d.reshape(*x.shape[:-1], hidden_size)

    launch_pdl = is_arch_support_pdl()
    use_single_kernel = (
        fuse_silu if fuse_silu is not None else (num_tokens <= _FUSED_UP_MAX_TOKENS)
    )
    if use_single_kernel:
        grid = (triton.cdiv(num_tokens, block_m) * (hidden_size // feats_per_tile),)
        _hc_mix_up_gate_kernel[grid](
            down_2d,
            w_up,
            x_2d,
            out_2d,
            num_tokens,
            down_2d.stride(0),
            down_2d.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            x_2d.stride(0),
            x_2d.stride(1),
            HIDDEN=hidden_size,
            RANK=lowrank,
            HC=hc_count,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            FUSE_SILU=True,
            LAUNCH_PDL=launch_pdl,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out_2d.reshape(*x.shape[:-1], hidden_size)

    act_2d = down_2d.new_empty((num_tokens, lowrank))
    _hc_silu_kernel[(num_tokens,)](
        down_2d,
        act_2d,
        num_tokens,
        down_2d.stride(0),
        act_2d.stride(0),
        RANK=lowrank,
        HC=hc_count,
        BLOCK_SIZE=triton.next_power_of_2(lowrank),
        LAUNCH_PDL=launch_pdl,
        num_warps=4,
    )
    gate_2d = F.linear(act_2d, w_up)
    gate_block = 512 if hidden_size % 512 == 0 else 256
    _hc_gate_mix_kernel[(num_tokens, triton.cdiv(hidden_size, gate_block))](
        x_2d,
        gate_2d,
        out_2d,
        num_tokens,
        x_2d.stride(0),
        gate_2d.stride(0),
        out_2d.stride(0),
        HIDDEN=hidden_size,
        HC=hc_count,
        BLOCK_SIZE=gate_block,
        LAUNCH_PDL=launch_pdl,
        num_warps=4,
    )
    return out_2d.reshape(*x.shape[:-1], hidden_size)
