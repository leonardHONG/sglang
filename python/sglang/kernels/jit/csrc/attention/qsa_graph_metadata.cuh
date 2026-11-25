#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr int32_t kQsaDraftMaxSteps = 16;
constexpr int64_t kQsaDraftStepPtrs = 9;
constexpr uint32_t kQsaPageThreads = 128;

SGL_DEVICE int64_t qsa_load_idx(const void* __restrict__ ptr, int64_t offset, bool is_i64) {
  if (is_i64) {
    return static_cast<const int64_t*>(ptr)[offset];
  }
  return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[offset]);
}

struct QsaGraphLayoutParams {
  const void* seq_lens;
  const void* req_pool;
  const void* extend_lens;
  int32_t* row_seq_lens;
  int32_t* row_prefix_lens;
  int32_t* row_req_pool;
  int32_t bs;
  int32_t num_tokens;
  int32_t num_padding;
  bool seq_is_i64;
  bool req_is_i64;
  bool ext_is_i64;
};

__global__ void qsa_graph_layout_kernel(const __grid_constant__ QsaGraphLayoutParams params) {
  const int32_t pid = static_cast<int32_t>(blockIdx.x);
  const int32_t real_reqs = params.bs - params.num_padding;

  if (pid == params.bs) {
    int32_t row_start = 0;
    for (int32_t j = 0; j < real_reqs; ++j) {
      row_start += static_cast<int32_t>(qsa_load_idx(params.extend_lens, j, params.ext_is_i64));
    }
    for (int32_t row = row_start + static_cast<int32_t>(threadIdx.x); row < params.num_tokens;
         row += static_cast<int32_t>(blockDim.x)) {
      params.row_seq_lens[row] = 1;
      params.row_prefix_lens[row] = 0;
      params.row_req_pool[row] = 0;
    }
    return;
  }

  if (threadIdx.x != 0) {
    return;
  }

  int32_t eff = 0;
  int32_t offset = 0;
  for (int32_t j = 0; j < real_reqs; ++j) {
    const int32_t e_j = static_cast<int32_t>(qsa_load_idx(params.extend_lens, j, params.ext_is_i64));
    if (j < pid) {
      offset += e_j;
    } else if (j == pid) {
      eff = e_j;
    }
  }
  if (eff <= 0) {
    return;
  }
  const int32_t base = static_cast<int32_t>(qsa_load_idx(params.seq_lens, pid, params.seq_is_i64));
  const int32_t req = static_cast<int32_t>(qsa_load_idx(params.req_pool, pid, params.req_is_i64));
  const int32_t prefix = base > eff ? (base - eff) : 0;
  for (int32_t j = 0; j < eff; ++j) {
    const int32_t row = offset + j;
    const int32_t cand = prefix + 1 + j;
    params.row_seq_lens[row] = cand < base ? cand : base;
    params.row_prefix_lens[row] = prefix;
    params.row_req_pool[row] = req;
  }
}

SGL_DEVICE void qsa_fill_row_metadata(
    int32_t* __restrict__ row_seq_lens,
    int32_t* __restrict__ row_req_pool,
    int32_t* __restrict__ compressed_lens,
    int32_t* __restrict__ write_locs,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ logical_positions,
    int64_t* __restrict__ state_slots,
    int32_t* __restrict__ ring_locs,
    const void* __restrict__ req_to_token,
    int64_t req_to_token_row_stride,
    bool r2t_is_i64,
    int32_t max_pages,
    int32_t ratio,
    int32_t full_page,
    const void* __restrict__ seq_lens,
    bool seq_is_i64,
    const void* __restrict__ req_pool,
    bool req_is_i64,
    int32_t* __restrict__ row_prefix_lens,
    int32_t bs,
    int32_t num_padding,
    int32_t extend_len,
    int32_t seq_offset,
    int32_t mode,
    int32_t row,
    int32_t page_tile,
    int32_t tid) {
  int32_t seq_len = 1;
  int64_t req = 0;
  int32_t prefix = 0;

  if (mode == 0 || mode == 1) {
    const int32_t req_row = (mode == 0) ? row : (row / extend_len);
    const bool real = req_row < (bs - num_padding);
    const int32_t base = real ? static_cast<int32_t>(qsa_load_idx(seq_lens, req_row, seq_is_i64)) : 0;
    req = real ? qsa_load_idx(req_pool, req_row, req_is_i64) : 0;
    if (mode == 0) {
      seq_len = real ? (base + seq_offset) : 1;
      prefix = seq_len > 1 ? (seq_len - 1) : 0;
    } else {
      seq_len = real ? (base + (row % extend_len) + 1) : 1;
      prefix = real ? base : 0;
    }
    if (page_tile == 0 && tid == 0) {
      row_seq_lens[row] = seq_len;
      row_req_pool[row] = static_cast<int32_t>(req);
      row_prefix_lens[row] = prefix;
    }
  } else {
    seq_len = row_seq_lens[row];
    req = static_cast<int64_t>(row_req_pool[row]);
  }

  const int64_t token_row = req * req_to_token_row_stride;
  if (page_tile == 0 && tid == 0) {
    const int32_t current = seq_len > 1 ? (seq_len - 1) : 0;
    const int32_t last_loc =
        static_cast<int32_t>(qsa_load_idx(req_to_token, token_row + current, r2t_is_i64));

    compressed_lens[row] = seq_len / ratio;
    const bool boundary = (seq_len > 0) && ((seq_len % ratio) == 0);
    write_locs[row] = boundary ? (last_loc / ratio) : 0;

    logical_positions[row] = current;
    state_slots[row] = req * static_cast<int64_t>(ratio) + static_cast<int64_t>(current % ratio);
    const int64_t ring_base = static_cast<int64_t>(row) * static_cast<int64_t>(ratio);
    for (int32_t k = 0; k < ratio; ++k) {
      const int32_t member = current > (ratio - 1 - k) ? (current - (ratio - 1 - k)) : 0;
      const int64_t slot = req * static_cast<int64_t>(ratio) + static_cast<int64_t>(member % ratio);
      ring_locs[ring_base + k] = static_cast<int32_t>(slot);
    }
  }

  const int64_t idx = static_cast<int64_t>(page_tile) * static_cast<int64_t>(kQsaPageThreads) + tid;
  const int64_t row_width_pages = req_to_token_row_stride / static_cast<int64_t>(full_page);
  const int64_t limit = max_pages < row_width_pages ? static_cast<int64_t>(max_pages) : row_width_pages;
  if (idx < limit) {
    const int64_t loc = qsa_load_idx(req_to_token, token_row + idx * static_cast<int64_t>(full_page), r2t_is_i64);
    const int64_t pg = loc / static_cast<int64_t>(full_page);
    page_table[static_cast<int64_t>(row) * static_cast<int64_t>(max_pages) + idx] =
        static_cast<int32_t>(pg > 0 ? pg : 0);
  }
}

struct QsaGraphRowMetadataParams {
  int32_t* row_seq_lens;
  int32_t* row_req_pool;
  int32_t* compressed_lens;
  int32_t* write_locs;
  int32_t* page_table;
  int32_t* logical_positions;
  int64_t* state_slots;
  int32_t* ring_locs;
  const void* req_to_token;
  const void* seq_lens;
  const void* req_pool;
  int32_t* row_prefix_lens;
  int64_t req_to_token_row_stride;
  int32_t max_pages;
  int32_t ratio;
  int32_t full_page;
  int32_t bs;
  int32_t num_padding;
  int32_t extend_len;
  int32_t seq_offset;
  int32_t mode;
  bool r2t_is_i64;
  bool seq_is_i64;
  bool req_is_i64;
};

__global__ void qsa_graph_row_metadata_kernel(const __grid_constant__ QsaGraphRowMetadataParams params) {
  qsa_fill_row_metadata(
      params.row_seq_lens,
      params.row_req_pool,
      params.compressed_lens,
      params.write_locs,
      params.page_table,
      params.logical_positions,
      params.state_slots,
      params.ring_locs,
      params.req_to_token,
      params.req_to_token_row_stride,
      params.r2t_is_i64,
      params.max_pages,
      params.ratio,
      params.full_page,
      params.seq_lens,
      params.seq_is_i64,
      params.req_pool,
      params.req_is_i64,
      params.row_prefix_lens,
      params.bs,
      params.num_padding,
      params.extend_len,
      params.seq_offset,
      params.mode,
      static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(blockIdx.y),
      static_cast<int32_t>(threadIdx.x));
}

struct QsaDraftStepBufs {
  int32_t* row_seq_lens;
  int32_t* row_req_pool;
  int32_t* compressed_lens;
  int32_t* write_locs;
  int32_t* page_table;
  int32_t* logical_positions;
  int64_t* state_slots;
  int32_t* ring_locs;
  int32_t* row_prefix_lens;
};

struct QsaDraftGraphMetadataParams {
  QsaDraftStepBufs steps[kQsaDraftMaxSteps];
  const void* req_to_token;
  const void* seq_lens;
  const void* req_pool;
  int64_t req_to_token_row_stride;
  int32_t max_pages;
  int32_t ratio;
  int32_t full_page;
  int32_t bs;
  int32_t num_padding;
  int32_t num_steps;
  bool r2t_is_i64;
  bool seq_is_i64;
  bool req_is_i64;
};

__global__ void qsa_draft_graph_metadata_kernel(const __grid_constant__ QsaDraftGraphMetadataParams params) {
  const int32_t step = static_cast<int32_t>(blockIdx.z);
  if (step >= params.num_steps) {
    return;
  }
  const QsaDraftStepBufs& b = params.steps[step];
  qsa_fill_row_metadata(
      b.row_seq_lens,
      b.row_req_pool,
      b.compressed_lens,
      b.write_locs,
      b.page_table,
      b.logical_positions,
      b.state_slots,
      b.ring_locs,
      params.req_to_token,
      params.req_to_token_row_stride,
      params.r2t_is_i64,
      params.max_pages,
      params.ratio,
      params.full_page,
      params.seq_lens,
      params.seq_is_i64,
      params.req_pool,
      params.req_is_i64,
      b.row_prefix_lens,
      params.bs,
      params.num_padding,
      0,
      step + 1,
      0,
      static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(blockIdx.y),
      static_cast<int32_t>(threadIdx.x));
}

struct QsaGraphMetadata {
  static void run_graph_metadata(
      const tvm::ffi::TensorView row_seq_lens,
      const tvm::ffi::TensorView row_req_pool,
      const tvm::ffi::TensorView row_prefix_lens,
      const tvm::ffi::TensorView compressed_lens,
      const tvm::ffi::TensorView write_locs,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView logical_positions,
      const tvm::ffi::TensorView state_slots,
      const tvm::ffi::TensorView ring_locs,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView req_pool,
      const tvm::ffi::TensorView extend_lens,
      int64_t mode,
      int64_t bs,
      int64_t num_rows,
      int64_t num_padding,
      int64_t extend_len,
      int64_t seq_offset,
      int64_t ratio,
      int64_t full_page) {
    using namespace host;
    if (bs <= 0 || num_rows <= 0) {
      return;
    }
    auto dev = SymbolicDevice{};
    dev.set_options<kDLGPU>();
    auto rows_sym = SymbolicSize{"num_rows"};
    rows_sym.set_value(num_rows);
    auto max_pages_sym = SymbolicSize{"max_pages"};
    auto ratio_sym = SymbolicSize{"ratio"};
    ratio_sym.set_value(ratio);

    TensorMatcher({rows_sym})  //
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(dev)
        .verify(row_seq_lens)
        .verify(row_req_pool)
        .verify(row_prefix_lens)
        .verify(compressed_lens)
        .verify(write_locs)
        .verify(logical_positions);
    TensorMatcher({rows_sym})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(state_slots);
    TensorMatcher({rows_sym, max_pages_sym})  //
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(dev)
        .verify(page_table);
    TensorMatcher({rows_sym, ratio_sym})  //
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(dev)
        .verify(ring_locs);
    TensorMatcher({-1, -1})  //
        .with_strides({-1, 1})
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(req_to_token);
    TensorMatcher({-1})  //
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(seq_lens);
    TensorMatcher({-1})  //
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(req_pool);
    TensorMatcher({-1})  //
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(extend_lens);

    CHECK_HOST(ratio > 0 && full_page > 0) << "Invalid QSA ratio/full_page: " << ratio << ", " << full_page;
    CHECK_HOST(mode == 0 || mode == 1 || mode == 2) << "Invalid QSA graph metadata mode: " << mode;
    if (mode == 1) {
      CHECK_HOST(extend_len > 0) << "extend_len must be positive in verify mode";
    }

    const DLDevice device = dev.unwrap();
    const bool r2t_is_i64 = is_type<int64_t>(req_to_token.dtype());
    const bool seq_is_i64 = is_type<int64_t>(seq_lens.dtype());
    const bool req_is_i64 = is_type<int64_t>(req_pool.dtype());
    const bool ext_is_i64 = is_type<int64_t>(extend_lens.dtype());
    const int32_t max_pages = static_cast<int32_t>(max_pages_sym.unwrap());

    if (mode == 2) {
      QsaGraphLayoutParams lp{};
      lp.seq_lens = seq_lens.data_ptr();
      lp.req_pool = req_pool.data_ptr();
      lp.extend_lens = extend_lens.data_ptr();
      lp.row_seq_lens = static_cast<int32_t*>(row_seq_lens.data_ptr());
      lp.row_prefix_lens = static_cast<int32_t*>(row_prefix_lens.data_ptr());
      lp.row_req_pool = static_cast<int32_t*>(row_req_pool.data_ptr());
      lp.bs = static_cast<int32_t>(bs);
      lp.num_tokens = static_cast<int32_t>(num_rows);
      lp.num_padding = static_cast<int32_t>(num_padding);
      lp.seq_is_i64 = seq_is_i64;
      lp.req_is_i64 = req_is_i64;
      lp.ext_is_i64 = ext_is_i64;
      LaunchKernel(dim3(static_cast<uint32_t>(bs + 1)), dim3(32), device)(qsa_graph_layout_kernel, lp);
    }

    QsaGraphRowMetadataParams rp{};
    rp.row_seq_lens = static_cast<int32_t*>(row_seq_lens.data_ptr());
    rp.row_req_pool = static_cast<int32_t*>(row_req_pool.data_ptr());
    rp.compressed_lens = static_cast<int32_t*>(compressed_lens.data_ptr());
    rp.write_locs = static_cast<int32_t*>(write_locs.data_ptr());
    rp.page_table = static_cast<int32_t*>(page_table.data_ptr());
    rp.logical_positions = static_cast<int32_t*>(logical_positions.data_ptr());
    rp.state_slots = static_cast<int64_t*>(state_slots.data_ptr());
    rp.ring_locs = static_cast<int32_t*>(ring_locs.data_ptr());
    rp.req_to_token = req_to_token.data_ptr();
    rp.seq_lens = seq_lens.data_ptr();
    rp.req_pool = req_pool.data_ptr();
    rp.row_prefix_lens = static_cast<int32_t*>(row_prefix_lens.data_ptr());
    rp.req_to_token_row_stride = req_to_token.stride(0);
    rp.max_pages = max_pages;
    rp.ratio = static_cast<int32_t>(ratio);
    rp.full_page = static_cast<int32_t>(full_page);
    rp.bs = static_cast<int32_t>(bs);
    rp.num_padding = static_cast<int32_t>(num_padding);
    rp.extend_len = static_cast<int32_t>(extend_len);
    rp.seq_offset = static_cast<int32_t>(seq_offset);
    rp.mode = static_cast<int32_t>(mode);
    rp.r2t_is_i64 = r2t_is_i64;
    rp.seq_is_i64 = seq_is_i64;
    rp.req_is_i64 = req_is_i64;

    const uint32_t page_tiles =
        static_cast<uint32_t>(div_ceil(static_cast<int64_t>(max_pages), static_cast<int64_t>(kQsaPageThreads)));
    LaunchKernel(dim3(static_cast<uint32_t>(num_rows), page_tiles), dim3(kQsaPageThreads), device)(
        qsa_graph_row_metadata_kernel, rp);
  }

  static void run_draft_graph_metadata(
      const tvm::ffi::TensorView step_ptrs_host,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView req_pool,
      int64_t bs,
      int64_t num_padding,
      int64_t max_pages,
      int64_t ratio,
      int64_t full_page) {
    using namespace host;
    if (bs <= 0) {
      return;
    }
    auto dev = SymbolicDevice{};
    dev.set_options<kDLGPU>();
    auto steps_sym = SymbolicSize{"num_steps"};
    auto ptrs_sym = SymbolicSize{"step_ptrs"};
    ptrs_sym.set_value(kQsaDraftStepPtrs);

    TensorMatcher({steps_sym, ptrs_sym})  //
        .with_dtype<int64_t>()
        .with_device<kDLCPU>()
        .verify(step_ptrs_host);
    TensorMatcher({-1, -1})  //
        .with_strides({-1, 1})
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(req_to_token);
    TensorMatcher({-1})  //
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(seq_lens);
    TensorMatcher({-1})  //
        .with_dtype<int32_t, int64_t>()
        .with_device<kDLGPU>(dev)
        .verify(req_pool);

    const int64_t num_steps = steps_sym.unwrap();
    if (num_steps <= 0) {
      return;
    }
    CHECK_HOST(num_steps <= kQsaDraftMaxSteps)
        << "num_steps exceeds kQsaDraftMaxSteps (" << num_steps << " > " << kQsaDraftMaxSteps << ")";
    CHECK_HOST(ratio > 0 && full_page > 0 && max_pages > 0)
        << "Invalid QSA draft params: ratio=" << ratio << " full_page=" << full_page << " max_pages=" << max_pages;

    QsaDraftGraphMetadataParams dp{};
    const int64_t* raw = static_cast<const int64_t*>(step_ptrs_host.data_ptr());
    for (int64_t s = 0; s < num_steps; ++s) {
      const int64_t* row = raw + s * kQsaDraftStepPtrs;
      dp.steps[s].row_seq_lens = reinterpret_cast<int32_t*>(row[0]);
      dp.steps[s].row_req_pool = reinterpret_cast<int32_t*>(row[1]);
      dp.steps[s].compressed_lens = reinterpret_cast<int32_t*>(row[2]);
      dp.steps[s].write_locs = reinterpret_cast<int32_t*>(row[3]);
      dp.steps[s].page_table = reinterpret_cast<int32_t*>(row[4]);
      dp.steps[s].logical_positions = reinterpret_cast<int32_t*>(row[5]);
      dp.steps[s].state_slots = reinterpret_cast<int64_t*>(row[6]);
      dp.steps[s].ring_locs = reinterpret_cast<int32_t*>(row[7]);
      dp.steps[s].row_prefix_lens = reinterpret_cast<int32_t*>(row[8]);
    }
    dp.req_to_token = req_to_token.data_ptr();
    dp.seq_lens = seq_lens.data_ptr();
    dp.req_pool = req_pool.data_ptr();
    dp.req_to_token_row_stride = req_to_token.stride(0);
    dp.max_pages = static_cast<int32_t>(max_pages);
    dp.ratio = static_cast<int32_t>(ratio);
    dp.full_page = static_cast<int32_t>(full_page);
    dp.bs = static_cast<int32_t>(bs);
    dp.num_padding = static_cast<int32_t>(num_padding);
    dp.num_steps = static_cast<int32_t>(num_steps);
    dp.r2t_is_i64 = is_type<int64_t>(req_to_token.dtype());
    dp.seq_is_i64 = is_type<int64_t>(seq_lens.dtype());
    dp.req_is_i64 = is_type<int64_t>(req_pool.dtype());

    const uint32_t page_tiles =
        static_cast<uint32_t>(div_ceil(max_pages, static_cast<int64_t>(kQsaPageThreads)));
    LaunchKernel(
        dim3(static_cast<uint32_t>(bs), page_tiles, static_cast<uint32_t>(num_steps)),
        dim3(kQsaPageThreads),
        dev.unwrap())(qsa_draft_graph_metadata_kernel, dp);
  }
};

}  // namespace sglang
