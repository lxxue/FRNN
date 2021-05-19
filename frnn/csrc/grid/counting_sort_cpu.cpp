#include <ATen/ATen.h>

void CountingSortCPU(const at::Tensor points, const at::Tensor lengths,
                     const at::Tensor grid_cell, const at::Tensor grid_idx,
                     const at::Tensor grid_off, at::Tensor sorted_points,
                     at::Tensor sorted_point_idx) {

  auto points_a = points.accessor<float, 3>();
  auto lengths_a = lengths.accessor<long, 1>();
  auto grid_cell_a = grid_cell.accessor<int, 2>();
  auto grid_idx_a = grid_idx.accessor<int, 2>();
  auto grid_off_a = grid_off.accessor<int, 2>();
  auto sorted_points_a = sorted_points.accessor<float, 3>();
  auto sorted_point_idx_a = sorted_point_idx.accessor<int, 2>();

  int N = points.size(0);
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < lengths_a[n]; ++p) {
      int cell_idx = grid_cell_a[n][p];
      int idx = grid_idx_a[n][p];
      int sorted_idx = grid_off_a[n][cell_idx] + idx;

      sorted_points_a[n][sorted_idx][0] = points_a[n][p][0];
      sorted_points_a[n][sorted_idx][1] = points_a[n][p][1];
      sorted_points_a[n][sorted_idx][2] = points_a[n][p][2];
      sorted_point_idx_a[n][sorted_idx] = p;
    }
  }
}