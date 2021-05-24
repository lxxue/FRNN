#include "grid/prefix_sum.h"
#include "grid/insert_points.h"
#include <ATen/ATen.h>

// a simple CPU prefix
at::Tensor PrefixSumCPU(const at::Tensor grid_cnt, const GridParams *params) {
  int N = grid_cnt.size(0);
  int G = grid_cnt.size(1);

  auto grid_cnt_a = grid_cnt.accessor<int, 2>();

  at::Tensor grid_off = at::full({N, G}, -1, grid_cnt.options());
  auto grid_off_a = grid_off.accessor<int, 2>();
  for (int n = 0; n < N; ++n) {
    grid_off_a[n][0] = 0;
    for (int p = 1; p < params[n].grid_total; ++p) {
      grid_off_a[n][p] = grid_off_a[n][p - 1] + grid_cnt_a[n][p - 1];
    }
  }
  return grid_off;
}

at::Tensor TestPrefixSumCPU(const at::Tensor bboxes, const at::Tensor points,
                            const at::Tensor lengths, float r) {
  int N = bboxes.size(0);
  int P = points.size(1);
  float cell_size = r;
  GridParams *h_params = new GridParams[N];
  int max_grid_total = 0;
  for (int n = 0; n < N; ++n) {
    SetupGridParams(bboxes.contiguous().data_ptr<float>() + n * 6, cell_size,
                    &h_params[n]);
    max_grid_total = std::max(max_grid_total, h_params[n].grid_total);
  }

  auto int_dtype = lengths.options().dtype(at::kInt);

  auto grid = at::full({N, max_grid_total}, -1, int_dtype);
  auto grid_cell = at::full({N, P}, -1, int_dtype);
  auto grid_cnt = at::zeros({N, max_grid_total}, int_dtype);
  auto grid_next = at::full({N, P}, -1, int_dtype);
  auto grid_idx = at::full({N, P}, -1, int_dtype);

  InsertPointsCPU(points, lengths, grid, grid_cnt, grid_cell, grid_next,
                  grid_idx, h_params);

  auto grid_off = PrefixSumCPU(grid_cnt, h_params);

  delete[] h_params;
  return grid_off;
}