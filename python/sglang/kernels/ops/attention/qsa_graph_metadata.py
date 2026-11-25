"""JIT CUDA builders for QSA CUDA-graph replay metadata.

Eliminates per-batch-size Triton JIT recompilations when decode/verify/draft_extend
batch sizes or padding counts fluctuate during serving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_DRAFT_STEPS = 16


@cache_once
def _jit_qsa_graph_metadata_module() -> Module:
    return load_jit(
        "qsa_graph_metadata",
        cuda_files=["attention/qsa_graph_metadata.cuh"],
        cuda_wrappers=[
            ("qsa_graph_metadata", "QsaGraphMetadata::run_graph_metadata"),
            ("qsa_draft_graph_metadata", "QsaGraphMetadata::run_draft_graph_metadata"),
        ],
    )


def can_use_jit_qsa_graph_metadata(device: torch.device) -> bool:
    return (
        torch.device(device).type == "cuda"
        and torch.version.hip is None
        and torch.cuda.is_available()
    )


def run_jit_qsa_graph_metadata(
    *,
    row_seq_lens: torch.Tensor,
    row_req_pool: torch.Tensor,
    row_prefix_lens: torch.Tensor,
    compressed_lens: torch.Tensor,
    write_locs: torch.Tensor,
    page_table: torch.Tensor,
    logical_positions: torch.Tensor,
    state_slots: torch.Tensor,
    ring_locs: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool: torch.Tensor,
    extend_lens: torch.Tensor,
    mode: int,
    bs: int,
    num_rows: int,
    num_padding: int,
    extend_len: int,
    seq_offset: int,
    ratio: int,
    full_page: int,
) -> None:
    module = _jit_qsa_graph_metadata_module()
    module.qsa_graph_metadata(
        row_seq_lens,
        row_req_pool,
        row_prefix_lens,
        compressed_lens,
        write_locs,
        page_table,
        logical_positions,
        state_slots,
        ring_locs,
        req_to_token,
        seq_lens,
        req_pool,
        extend_lens,
        int(mode),
        int(bs),
        int(num_rows),
        int(num_padding),
        int(extend_len),
        int(seq_offset),
        int(ratio),
        int(full_page),
    )


def pack_draft_step_ptrs_host(buffers: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor | None:
    if not buffers or len(buffers) > _MAX_DRAFT_STEPS:
        return None
    rows = []
    for step_bufs in buffers:
        if len(step_bufs) != 9:
            return None
        # Verify expected contiguous dtypes match the CUDA struct contract.
        if (
            step_bufs[0].dtype != torch.int32
            or step_bufs[1].dtype != torch.int32
            or step_bufs[2].dtype != torch.int32
            or step_bufs[3].dtype != torch.int32
            or step_bufs[4].dtype != torch.int32
            or step_bufs[5].dtype != torch.int32
            or step_bufs[6].dtype != torch.int64
            or step_bufs[7].dtype != torch.int32
            or step_bufs[8].dtype != torch.int32
        ):
            return None
        if not all(t.is_contiguous() and t.is_cuda for t in step_bufs):
            return None
        rows.append([t.data_ptr() for t in step_bufs])
    return torch.tensor(rows, dtype=torch.int64)


def run_jit_qsa_draft_graph_metadata(
    *,
    step_ptrs_host: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool: torch.Tensor,
    bs: int,
    num_padding: int,
    max_pages: int,
    ratio: int,
    full_page: int,
) -> None:
    module = _jit_qsa_graph_metadata_module()
    module.qsa_draft_graph_metadata(
        step_ptrs_host,
        req_to_token,
        seq_lens,
        req_pool,
        int(bs),
        int(num_padding),
        int(max_pages),
        int(ratio),
        int(full_page),
    )


__all__ = [
    "can_use_jit_qsa_graph_metadata",
    "pack_draft_step_ptrs_host",
    "run_jit_qsa_draft_graph_metadata",
    "run_jit_qsa_graph_metadata",
]
