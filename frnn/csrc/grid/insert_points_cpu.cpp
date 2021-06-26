#include "grid/insert_points.h"

int GetGridCell(float x, float y, float z, int3 &gc, GridParams &params) {
  gc.x = (int)((x - params.grid_min.x) * params.grid_delta);
  gc.y = (int)((y - params.grid_min.y) * params.grid_delta);
  gc.z = (int)((z - params.grid_min.z) * params.grid_delta);

  return (gc.x * params.grid_res.y + gc.y) * params.grid_res.z + gc.z;
}

void InsertPointsCPU(const at::Tensor points, const at::Tensor lengths,
                     at::Tensor grid, at::Tensor grid_cnt, at::Tensor grid_cell,
                     at::Tensor grid_next, at::Tensor grid_idx,
                     GridParams *params) {
  auto points_a = points.accessor<float, 3>();
  auto lengths_a = lengths.accessor<long, 1>();
  auto grid_a = grid.accessor<int, 2>();
  auto grid_cnt_a = grid_cnt.accessor<int, 2>();
  auto grid_cell_a = grid_cell.accessor<int, 2>();
  auto grid_next_a = grid_next.accessor<int, 2>();
  auto grid_idx_a = grid_idx.accessor<int, 2>();

  int gs;
  int3 gc;
  int N = points.size(0);
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < lengths_a[n]; ++p) {
      gs = GetGridCell(points_a[n][p][0], points_a[n][p][1], points_a[n][p][2],
                       gc, params[n]);
      grid_cell_a[n][p] = gs;
      grid_next_a[n][p] = grid_a[n][gs];
      grid_idx_a[n][p] = grid_cnt_a[n][gs];
      grid_a[n][gs] = p;
      grid_cnt_a[n][gs]++;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCPU(
    const at::Tensor bboxes, const at::Tensor points, const at::Tensor lengths,
    float r) {
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

  delete[] h_params;
  return std::make_tuple(grid_cnt, grid_cell, grid_idx);
}
