#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>
#include <queue>

#include "grid.h"

int GetGridCell(float x, float y, float z, int3& gc, GridParams& params) {
  gc.x = (int) ((x - params.grid_min.x) * params.grid_delta);
  gc.y = (int) ((y - params.grid_min.y) * params.grid_delta);
  gc.z = (int) ((z - params.grid_min.z) * params.grid_delta);

  return (gc.x*params.grid_res.y + gc.y)*params.grid_res.z + gc.z;
}

void InsertPointsCPU(
    const at::Tensor points, 
    const at::Tensor lengths, 
    at::Tensor grid, 
    at::Tensor grid_cnt, 
    at::Tensor grid_cell, 
    at::Tensor grid_next, 
    at::Tensor grid_idx,
    GridParams* params) {
  auto points_a = points.accessor<float, 3>();
  auto lengths_a = lengths.accessor<int, 1>();
  auto grid_a = grid.accessor<int, 2>();
  auto grid_cnt_a = grid_cnt.accessor<int, 2>();
  auto grid_cell_a = grid_cell.accessor<int, 2>();
  auto grid_next_a = grid_next.accessor<int, 2>();
  auto grid_idx_a = grid_idx.accessor<int, 2>();

  int gs;
  int3 gc;
  int N = points.size(0);
  for (int n=0; n < N; ++n) {
    for (int p=0; p < lengths_a[n]; ++p) {
      gs = GetGridCell(points_a[n][p][0], points_a[n][p][1], points_a[n][p][2], gc, params[n]);
      grid_cell_a[n][p] = gs;
      grid_next_a[n][p] = grid_a[n][gs];
      grid_idx_a[n][p] = grid_cnt_a[n][gs];
      grid_a[n][gs] = p;
      grid_cnt_a[n][gs]++;
    }
  }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCPU(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r) {
  int N = bboxes.size(0);
  int P = points.size(1);
  float cell_size = r;
  GridParams* h_params = new GridParams[N];
  int max_grid_total = 0;
  for (int n = 0; n < N; ++n) {
    SetupGridParams(
      bboxes.contiguous().data_ptr<float>() + n*6,
      cell_size,
      &h_params[n]
    );
    max_grid_total = std::max(max_grid_total, h_params[n].grid_total);
  }

  auto int_dtype = lengths.options().dtype(at::kInt);
  auto dtype = int_dtype;

  auto grid = at::full({N, max_grid_total}, -1, dtype);
  auto grid_cell = at::full({N, P}, -1, dtype); 
  auto grid_cnt = at::zeros({N, max_grid_total}, dtype);
  auto grid_next = at::full({N, P}, -1, dtype); 
  auto grid_idx = at::full({N, P}, -1, dtype); 

  InsertPointsCPU(
    points,
    lengths,
    grid,
    grid_cnt,
    grid_cell,
    grid_next,
    grid_idx,
    h_params
  );

  delete[] h_params;
  return std::make_tuple(grid_cnt, grid_cell, grid_idx);
}

std::tuple<at::Tensor, at::Tensor> FindNbrsCPU(
    const at::Tensor points1,             // (N, P1, 3)
    const at::Tensor points2,             // (N, P2, 3)
    const at::Tensor lengths1,            // (N,)
    const at::Tensor lengths2,            // (N,)
    const at::Tensor grid_off,            // (N, G)
    const at::Tensor sorted_point_idx,    // (N, P2)
    const GridParams* params,
    int K,
    float r) {
  
  const int N = points1.size(0);
  // const int G = grid_off.size(1);
  float r2 = r*r;
  float3 diff;

  auto points1_a = points1.accessor<float, 3>();
  auto points2_a = points2.accessor<float, 3>();
  auto lengths1_a = lengths1.accessor<int, 1>();
  auto lengths2_a = lengths2.accessor<int, 1>();
  auto grid_off_a = grid_off.accessor<int, 2>();
  auto sorted_point_idx_a = sorted_point_idx.accessor<int, 2>();


  for (int n=0; n < N; ++n) {
    int3 res = params[n].grid_res;
    float3 grid_min = params[n].grid_min;
    float grid_delta = params[n].grid_delta;

    for (int p1=0; p1 < lengths1_a[n]; ++p1) {
      float3 cur_point;
      cur_point.x = points1_a[n][p1][0];
      cur_point.y = points1_a[n][p1][1];
      cur_point.z = points1_a[n][p1][2];
      int3 min_gc, max_gc;

      min_gc.x = (int) std::floor((cur_point.x-grid_min.x-r)*grid_delta);
      min_gc.y = (int) std::floor((cur_point.y-grid_min.y-r)*grid_delta);
      min_gc.z = (int) std::floor((cur_point.z-grid_min.z-r)*grid_delta);
      max_gc.x = (int) std::floor((cur_point.x-grid_min.x-r)*grid_delta);
      max_gc.y = (int) std::floor((cur_point.y-grid_min.y-r)*grid_delta);
      max_gc.z = (int) std::floor((cur_point.z-grid_min.z-r)*grid_delta);
      for (int x=std::max(min_gc.x, 0); x<=std::min(max_gc.x, res.x-1); ++x) {
        for (int y=std::max(min_gc.y, 0); y<=std::min(max_gc.y, res.y-1); ++y) {
          for (int z=std::max(min_gc.z, 0); z<=std::min(max_gc.z, res.z-1); ++z) {
            int cell_idx = (x*res.y + y)*res.z + z;
            int p2_start = grid_off_a[n][cell_idx];
            int p2_end;
            if (cell_idx+1 == params[n].grid_total) {
              p2_end = lengths2_a[n];
            }
            else {
              p2_end = grid_off_a[n][cell_idx+1];
            }
            for (int p2=p2_start; p2<p2_end; ++p2) {
              diff.x = points2_a[n][p2][0] - cur_point.x;
              diff.y = points2_a[n][p2][1] - cur_point.y;
              diff.z = points2_a[n][p2][2] - cur_point.z;
              float sqdist = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
              if (sqdist <= r2) {
              }
            }
          }
        }
      }
    }
  }
}