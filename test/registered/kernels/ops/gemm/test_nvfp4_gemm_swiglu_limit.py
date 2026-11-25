"""Correctness tests for SM100 fused NVFP4 GEMM + SwiGLU (+ swiglu_limit clamp) + NVFP4 quant."""

import sys

import pytest
import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = "cuda"
BLOCK_SIZE = 16
FP4_VALUE_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _sm10x_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _make_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    amax = tensor.abs().max().to(torch.float32).clamp(min=1e-6)
    return (448.0 * 6.0 / amax).to(device=tensor.device, dtype=torch.float32)


def _unpack_fp4_bytes(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(FP4_VALUE_LUT, dtype=torch.float32, device=packed.device)

    def _decode(nibbles: torch.Tensor) -> torch.Tensor:
        values = lut[(nibbles & 0x7).to(torch.long)]
        return torch.where((nibbles & 0x8) != 0, -values, values)

    low = _decode(packed & 0x0F)
    high = _decode((packed & 0xF0) >> 4)
    return torch.stack((low, high), dim=-1).reshape(
        packed.shape[0], packed.shape[1] * 2
    )


def _swizzled_to_linear(
    scales_swizzled: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scales_swizzled = scales_swizzled.view(torch.float8_e4m3fn)
    row_tiles = (rows + 128 - 1) // 128
    tile_cols = BLOCK_SIZE * 4
    col_tiles = (cols + tile_cols - 1) // tile_cols
    tmp = scales_swizzled.reshape(1, row_tiles, col_tiles, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    linear = tmp.reshape(row_tiles * 128, col_tiles * tile_cols // BLOCK_SIZE)
    return linear[:rows, : cols // BLOCK_SIZE]


def _dequantize_nvfp4(
    packed: torch.Tensor,
    scales_swizzled: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    unpacked = _unpack_fp4_bytes(packed).reshape(rows, cols // BLOCK_SIZE, BLOCK_SIZE)
    scales_linear = _swizzled_to_linear(scales_swizzled, rows, cols).to(torch.float32)
    return (unpacked * (scales_linear / global_scale).unsqueeze(-1)).reshape(rows, cols)


def _dequantize_nvfp4_linear_scale(
    packed: torch.Tensor,
    scales_linear: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    unpacked = _unpack_fp4_bytes(packed).reshape(rows, cols // BLOCK_SIZE, BLOCK_SIZE)
    scales = scales_linear.view(torch.float8_e4m3fn).to(torch.float32)
    return (unpacked * (scales / global_scale).unsqueeze(-1)).reshape(rows, cols)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1.0 - 2.0 * (x * y).sum() / denom).item()


@pytest.mark.skipif(
    not _sm10x_supported(),
    reason="nvfp4_gemm_swiglu_nvfp4_quant requires Blackwell SM10x",
)
@pytest.mark.parametrize("m", [64, 256], ids=["1cta_m64", "2cta_m256"])
@pytest.mark.parametrize("vectorized_f32", [True, False], ids=["vec_f32", "scalar_f32"])
@pytest.mark.parametrize("swiglu_limit", [None, 7.0, 10.0])
def test_nvfp4_gemm_swiglu_limit_matches_reference(
    m: int, vectorized_f32: bool, swiglu_limit: float | None
) -> None:
    """Verify fused NVFP4 gate_up + SwiGLU (+ optional clamp) + FP4 quant against
    unfused GEMM -> silu_and_mul_clamp -> fp4_quantize on inputs that exceed limit."""
    import flashinfer

    from sglang.kernels.ops.gemm.nvfp4_gemm_swiglu_nvfp4_quant import (
        interleave_linear_and_gate,
        nvfp4_gemm_swiglu_nvfp4_quant,
        swizzle_blockscale_2d,
    )

    k = 512
    inter = 256
    gen = torch.Generator(device=DEVICE).manual_seed(20261126 + m)

    # Scale inputs so pre-activation gate and up values routinely reach +-30,
    # strongly exceeding swiglu_limit in all four sign quadrants.
    x = torch.randn((m, k), device=DEVICE, dtype=torch.bfloat16, generator=gen) * 1.5
    w = (
        torch.randn((2 * inter, k), device=DEVICE, dtype=torch.bfloat16, generator=gen)
        * 0.6
    )

    x_gs = _make_global_scale(x)
    w_gs = _make_global_scale(w)
    alpha = (1.0 / (x_gs * w_gs)).to(torch.float32)

    x_fp4, x_sf = flashinfer.fp4_quantize(x, x_gs)
    if x_sf.dtype == torch.uint8:
        x_sf = x_sf.view(torch.float8_e4m3fn)

    w_fp4, w_sf_linear = flashinfer.fp4_quantize(w, w_gs, is_sf_swizzled_layout=False)
    if w_sf_linear.dtype == torch.uint8:
        w_sf_linear = w_sf_linear.view(torch.float8_e4m3fn)

    # Match ModelOptFp4LinearMethod.process_weights_after_loading interleaving.
    gate_w, up_w = w_fp4.chunk(2, dim=0)
    w_swiglu = interleave_linear_and_gate(
        torch.cat((up_w, gate_w), dim=0), group_size=64, dim=0
    )
    gate_s, up_s = w_sf_linear.chunk(2, dim=0)
    w_sf_swiglu = swizzle_blockscale_2d(
        interleave_linear_and_gate(
            torch.cat((up_s, gate_s), dim=0), group_size=64, dim=0
        )
    )

    # Exact unfused GEMM from the same quantized operands.
    x_dq = _dequantize_nvfp4(x_fp4, x_sf, x_gs)
    w_dq = _dequantize_nvfp4_linear_scale(w_fp4, w_sf_linear, w_gs)
    gate_up = torch.matmul(x_dq, w_dq.t())
    gate, up = gate_up.chunk(2, dim=-1)

    if swiglu_limit is not None:
        # Confirm the test actually exercises both sides of the clamp boundary.
        assert (gate > swiglu_limit).any()
        assert (gate < -swiglu_limit).any()
        assert (up > swiglu_limit).any()
        assert (up < -swiglu_limit).any()

        ref_act_bf16 = torch.empty((m, inter), device=DEVICE, dtype=torch.bfloat16)
        silu_and_mul_clamp(gate_up.to(torch.bfloat16), ref_act_bf16, swiglu_limit)
        ref_act_fp32 = torch.nn.functional.silu(
            gate.clamp(max=swiglu_limit)
        ) * up.clamp(min=-swiglu_limit, max=swiglu_limit)
    else:
        ref_act_fp32 = torch.nn.functional.silu(gate) * up
        ref_act_bf16 = ref_act_fp32.to(torch.bfloat16)

    out_gs = _make_global_scale(ref_act_fp32)

    out_fp4, out_sf = nvfp4_gemm_swiglu_nvfp4_quant(
        x_fp4,
        x_sf,
        w_swiglu,
        w_sf_swiglu,
        alpha,
        out_gs,
        vectorized_f32=vectorized_f32,
        enable_pdl=False,
        swiglu_limit=swiglu_limit,
    )
    got = _dequantize_nvfp4(out_fp4, out_sf, out_gs)

    ref_fp4, ref_sf = flashinfer.fp4_quantize(ref_act_bf16, out_gs)
    if ref_sf.dtype == torch.uint8:
        ref_sf = ref_sf.view(torch.float8_e4m3fn)
    expected = _dequantize_nvfp4(ref_fp4, ref_sf, out_gs)

    # Fused kernel clamps in fp32 before FP4 quant; reference clamps after bf16
    # rounding then quantizes to FP4. They must agree tightly.
    diff_vs_ref_quant = _calc_diff(got, expected)
    diff_vs_fp32 = _calc_diff(got, ref_act_fp32)
    assert diff_vs_ref_quant < 5e-3, (
        f"m={m}, vec={vectorized_f32}, limit={swiglu_limit}: "
        f"diff_vs_ref_quant={diff_vs_ref_quant:.6f}, diff_vs_fp32={diff_vs_fp32:.6f}"
    )
    assert diff_vs_fp32 < 1e-2, (
        f"m={m}, vec={vectorized_f32}, limit={swiglu_limit}: "
        f"diff_vs_fp32={diff_vs_fp32:.6f}"
    )

    if swiglu_limit is not None:
        # Clamping must bound output magnitude and diverge clearly from unclamped.
        max_clamped_theory = float(
            torch.nn.functional.silu(torch.tensor(swiglu_limit)) * swiglu_limit
        )
        assert got.abs().max().item() <= max_clamped_theory * 1.15

        unclamped_gs = _make_global_scale(torch.nn.functional.silu(gate) * up)
        unclamped_fp4, unclamped_sf = nvfp4_gemm_swiglu_nvfp4_quant(
            x_fp4,
            x_sf,
            w_swiglu,
            w_sf_swiglu,
            alpha,
            unclamped_gs,
            vectorized_f32=vectorized_f32,
            enable_pdl=False,
            swiglu_limit=None,
        )
        unclamped = _dequantize_nvfp4(unclamped_fp4, unclamped_sf, unclamped_gs)
        assert unclamped.abs().max().item() > got.abs().max().item() * 1.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
