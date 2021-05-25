#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <queue>
#include <tuple>

#include "grid/counting_sort.h"
#include "grid/find_nbrs.h"
#include "grid/grid.h"
#include "grid/insert_points.h"
#include "grid/prefix_sum.h"

std::tuple<at::Tensor, at::Tensor> FindNbrsCPU(
    const at::Tensor points1,           // (N, P1, 3)
    const at::Tensor points2,           // (N, P2, 3)
    const at::Tensor lengths1,          // (N,)
    const at::Tensor lengths2,          // (N,)
    const at::Tensor grid_off,          // (N, G)
    const at::Tensor sorted_point_idx,  // (N, P2)
    const GridParams *params, int K, float r) {
  int N = points1.size(0);
  int P1 = points1.size(1);
  // const int G = grid_off.size(1);
  float r2 = r * r;
  float3 diff;

  auto points1_a = points1.accessor<float, 3>();
  auto points2_a = points2.accessor<float, 3>();
  auto lengths1_a = lengths1.accessor<long, 1>();
  auto lengths2_a = lengths2.accessor<long, 1>();
  auto grid_off_a = grid_off.accessor<int, 2>();
  auto sorted_point_idx_a = sorted_point_idx.accessor<int, 2>();

  auto idxs = at::full({N, P1, K}, -1, lengths1.options());
  auto dists = at::full({N, P1, K}, -1, points1.options());

  auto idxs_a = idxs.accessor<long, 3>();
  auto dists_a = dists.accessor<float, 3>();

  for (int n = 0; n < N; ++n) {
    int3 res = params[n].grid_res;
    float3 grid_min = params[n].grid_min;
    float grid_delta = params[n].grid_delta;

    for (int p1 = 0; p1 < lengths1_a[n]; ++p1) {
      float3 cur_point;
      cur_point.x = points1_a[n][p1][0];
      cur_point.y = points1_a[n][p1][1];
      cur_point.z = points1_a[n][p1][2];
      int3 min_gc, max_gc;

      min_gc.x = (int)std::floor((cur_point.x - grid_min.x - r) * grid_delta);
      min_gc.y = (int)std::floor((cur_point.y - grid_min.y - r) * grid_delta);
      min_gc.z = (int)std::floor((cur_point.z - grid_min.z - r) * grid_delta);
      max_gc.x = (int)std::floor((cur_point.x - grid_min.x + r) * grid_delta);
      max_gc.y = (int)std::floor((cur_point.y - grid_min.y + r) * grid_delta);
      max_gc.z = (int)std::floor((cur_point.z - grid_min.z + r) * grid_delta);

      std::priority_queue<std::tuple<float, int>> q;
      for (int x = std::max(min_gc.x, 0); x <= std::min(max_gc.x, res.x - 1);
           ++x) {
        for (int y = std::max(min_gc.y, 0); y <= std::min(max_gc.y, res.y - 1);
             ++y) {
          for (int z = std::max(min_gc.z, 0);
               z <= std::min(max_gc.z, res.z - 1); ++z) {
            int cell_idx = (x * res.y + y) * res.z + z;
            int p2_start = grid_off_a[n][cell_idx];
            int p2_end;
            if (cell_idx + 1 == params[n].grid_total) {
              p2_end = lengths2_a[n];
            } else {
              p2_end = grid_off_a[n][cell_idx + 1];
            }
            for (int p2 = p2_start; p2 < p2_end; ++p2) {
              diff.x = points2_a[n][p2][0] - cur_point.x;
              diff.y = points2_a[n][p2][1] - cur_point.y;
              diff.z = points2_a[n][p2][2] - cur_point.z;
              float sqdist =
                  diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
              if (sqdist <= r2) {
                int size = static_cast<int>(q.size());
                if (size < K) {
                  q.emplace(sqdist, sorted_point_idx_a[n][p2]);
                } else if (sqdist < std::get<0>(q.top())) {
                  q.emplace(sqdist, sorted_point_idx_a[n][p2]);
                  q.pop();
                }
              }
            }
          }
        }
      }
      while (!q.empty()) {
        auto t = q.top();
        q.pop();
        const int k = q.size();
        dists_a[n][p1][k] = std::get<0>(t);
        idxs_a[n][p1][k] = std::get<1>(t);
      }
    }
  }
  return std::make_tuple(idxs, dists);
}

// deprecated cpp functions
void SetupGridParams(float *bboxes, float cell_size, GridParams *params) {
  /*
    params->grid_min.x = bboxes[0];
    params->grid_max.x = bboxes[1];
    params->grid_min.y = bboxes[2];
    params->grid_max.y = bboxes[3];
    params->grid_min.z = bboxes[4];
    params->grid_max.z = bboxes[5];

    params->grid_size = params->grid_max - params->grid_min;
    float res_min = std::min(std::min(params->grid_size.x, params->grid_size.y),
                             params->grid_size.z);
    if (cell_size < res_min / GRID_3D_MAX_RES)
      cell_size = res_min / GRID_3D_MAX_RES;
    params->grid_res.x = (int)(params->grid_size.x / cell_size) + 1;
    params->grid_res.y = (int)(params->grid_size.y / cell_size) + 1;
    params->grid_res.z = (int)(params->grid_size.z / cell_size) + 1;
    params->grid_total =
        params->grid_res.x * params->grid_res.y * params->grid_res.z;

    params->grid_delta = 1 / cell_size;

    return;
  */
}

std::tuple<at::Tensor, at::Tensor> TestFindNbrsCPU(
    const at::Tensor bboxes, const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2, int K, float r) {
  int N = points1.size(0);
  int P2 = points2.size(1);
  float cell_size = r;
  GridParams *h_params = new GridParams[N];
  int max_grid_total = 0;
  for (int i = 0; i < N; ++i) {
    SetupGridParams(bboxes.contiguous().data_ptr<float>() + i * 6, cell_size,
                    &h_params[i]);
    max_grid_total = std::max(max_grid_total, h_params[i].grid_total);
  }

  auto int_dtype = lengths2.options().dtype(at::kInt);

  auto grid = at::full({N, P2}, -1, int_dtype);
  auto grid_next = at::full({N, P2}, -1, int_dtype);
  auto grid_cnt = at::zeros({N, max_grid_total}, int_dtype);
  auto grid_cell = at::full({N, P2}, -1, int_dtype);
  auto grid_idx = at::full({N, P2}, -1, int_dtype);

  InsertPointsCPU(points2, lengths2, grid, grid_cnt, grid_cell, grid_next,
                  grid_idx, h_params);

  auto grid_off = PrefixSumCPU(grid_cnt, h_params);

  auto sorted_points2 = at::zeros({N, P2, 3}, points2.options());
  auto sorted_point_idx = at::full({N, P2}, -1, int_dtype);

  CountingSortCPU(points2, lengths2, grid_cell, grid_idx, grid_off,
                  sorted_points2, sorted_point_idx);

  auto results = FindNbrsCPU(points1, sorted_points2, lengths1, lengths2,
                             grid_off, sorted_point_idx, h_params, K, r);

  delete[] h_params;
  return results;
}