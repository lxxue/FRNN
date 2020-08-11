#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>

#include "grid.h"
#include "prefix_sum.h"
#include "counting_sort.h"
#include "utils/mink.cuh"
#include "utils/dispatch.h"


void SetupGridParams(
    float* bboxes,
    float cell_size,
    GridParams* params) {
  params->grid_min.x = bboxes[0];
  params->grid_max.x = bboxes[1];
  params->grid_min.y = bboxes[2];
  params->grid_max.y = bboxes[3];
  params->grid_min.z = bboxes[4];
  params->grid_max.z = bboxes[5];

  params->grid_size = params->grid_max - params->grid_min;
  float res_min = std::min(std::min(params->grid_size.x, params->grid_size.y), params->grid_size.z);
  if (cell_size < res_min/MAX_RES)
    cell_size = res_min / MAX_RES;
  params->grid_res.x = (int)(params->grid_size.x / cell_size) + 1;
  params->grid_res.y = (int)(params->grid_size.y / cell_size) + 1;
  params->grid_res.z = (int)(params->grid_size.z / cell_size) + 1;
  params->grid_total = params->grid_res.x * params->grid_res.y * params->grid_res.z;

  params->grid_delta = 1 / cell_size;

  return;
}
/*
void SetupGridParams(
    float* bboxes,
    float cell_size,
    float* params) {
  params[GRID_MIN_X] = bboxes[0];
  params[GRID_MIN_Y] = bboxes[2];
  params[GRID_MIN_Z] = bboxes[4];
  float grid_size_x = bboxes[1] - bboxes[0];
  float grid_size_y = bboxes[3] - bboxes[2];
  float grid_size_z = bboxes[5] - bboxes[4];

  float res_min = std::min(std::min(grid_size_x, grid_size_y), grid_size_z);
  if (cell_size < res_min/MAX_RES)
    cell_size = res_min / MAX_RES;
  params[GRID_RES_X] = (int)(grid_size_x / cell_size) + 1;
  params[GRID_RES_Y] = (int)(grid_size_y / cell_size) + 1;
  params[GRID_RES_Z] = (int)(grid_size_z / cell_size) + 1;
  params[GRID_TOTAL]= params[GRID_RES_X] * params[GRID_RES_Y] * params[GRID_RES_Z];

  params[GRID_DELTA] = 1 / cell_size;

  return;
}
*/

void TestSetupGridParamsCUDA(
    const at::Tensor bboxes,  // N x 3 x 2 (min, max) at last dimension
    float r) {
  int N = bboxes.size(0);
  // cell_size determined joint by search radius and bbox_size
  // cell_size different for different point clouds in the batch
  float cell_size = r;
  GridParams* h_params = new GridParams[N];
  for (int i = 0; i < N; ++i) {
    SetupGridParams(
      bboxes.contiguous().data_ptr<float>() + i*6,
      cell_size,
      &h_params[i]
    );
    // std::cout << h_params[i].grid_min.x << ' ' << h_params[i].grid_min.y << ' ' << h_params[i].grid_min.z << std::endl;
    // std::cout << h_params[i].grid_max.x << ' ' << h_params[i].grid_max.y << ' ' << h_params[i].grid_max.z << std::endl;
    // std::cout << h_params[i].grid_size.x << ' ' << h_params[i].grid_size.y << ' ' << h_params[i].grid_size.z << std::endl;
    // std::cout << h_params[i].grid_res.x << ' ' << h_params[i].grid_res.y << ' ' << h_params[i].grid_res.z << std::endl;
    // std::cout << h_params[i].grid_total << ' ' << h_params[i].grid_delta << ' ' << std::endl; 
  }

  GridParams* d_params;
  cudaMalloc((void**)&d_params, N*sizeof(GridParams));
  cudaMemcpy(d_params, h_params, N*sizeof(GridParams), cudaMemcpyHostToDevice);

  GridParams* h_d_params = new GridParams[N];
  cudaMemcpy(h_d_params, d_params, N*sizeof(GridParams), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) {
    std::cout << h_d_params[i].grid_min.x << ' ' << h_d_params[i].grid_min.y << ' ' << h_d_params[i].grid_min.z << std::endl;
    std::cout << h_d_params[i].grid_max.x << ' ' << h_d_params[i].grid_max.y << ' ' << h_d_params[i].grid_max.z << std::endl;
    std::cout << h_d_params[i].grid_res.x << ' ' << h_d_params[i].grid_res.y << ' ' << h_d_params[i].grid_res.z << std::endl;
    std::cout << h_d_params[i].grid_total << ' ' << h_d_params[i].grid_delta << ' ' << std::endl; 
    std::cout << h_d_params[i].grid_size.x << ' ' << h_d_params[i].grid_size.y << ' ' << h_d_params[i].grid_size.z << std::endl;
  }
  delete[] h_params;
  delete[] h_d_params;
  cudaFree(d_params);
  return;
}
/*
*/

__global__ void InsertPointsKernel(
    const float* __restrict__ points,
    const long* __restrict__ lengths,
    int* grid_cnt, // not sure if we can use __restrict__ here
    int* __restrict__ grid_cell,
    int* __restrict__ grid_idx,
    int N,
    int P,
    int G,
    const GridParams* params) {

  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk=blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n])
      continue;

    float3 grid_min = params[n].grid_min;
    float grid_delta = params[n].grid_delta;
    int3 grid_res = params[n].grid_res;

    int3 gc;
    gc.x = (int) ((points[(n*P+p)*3+0]-grid_min.x) * grid_delta);
    gc.y = (int) ((points[(n*P+p)*3+1]-grid_min.y) * grid_delta);
    gc.z = (int) ((points[(n*P+p)*3+2]-grid_min.z) * grid_delta);

    int gs = (gc.x*grid_res.y + gc.y) * grid_res.z + gc.z;
    grid_cell[n*P+p] = gs;
    // for long, need to convert it to unsigned long long since there is no atomicAdd for long
    // grid_idx[n * P + p] = atomicAdd((unsigned long long*)&grid_cnt[n*grid_total + gs], (unsigned long long)1);
    grid_idx[n*P+p] = atomicAdd(&grid_cnt[n*G + gs], 1);
  } 
}

void InsertPointsCUDA(
    const at::Tensor points,    // (N, P, 3)
    const at::Tensor lengths,   // (N,)
    at::Tensor grid_cnt,        // (N, G)
    at::Tensor grid_cell,       // (N, P)      
    at::Tensor grid_idx,        // (N, P)
    int G,
    const GridParams* params) { // (N,)
  
  at::TensorArg points_t{points, "points", 1};
  at::TensorArg lengths_t{lengths, "lengths", 2};
  at::TensorArg grid_cnt_t{grid_cnt, "grid_cnt", 3};
  at::TensorArg grid_cell_t{grid_cell, "grid_cell", 4};
  at::TensorArg grid_idx_t{grid_idx, "grid_idx", 5};

  at::CheckedFrom c = "InsertPointsCUDA";
  at::checkAllSameGPU(c, {points_t, lengths_t, grid_cnt_t, grid_cell_t, grid_idx_t});
  at::checkAllSameType(c, {grid_cnt_t, grid_cell_t, grid_idx_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = 256;

  InsertPointsKernel<<<blocks, threads, 0, stream>>>(
    points.contiguous().data_ptr<float>(),
    lengths.contiguous().data_ptr<long>(),
    grid_cnt.contiguous().data_ptr<int>(),
    grid_cell.contiguous().data_ptr<int>(),
    grid_idx.contiguous().data_ptr<int>(),
    points.size(0),
    points.size(1),
    G,
    params
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCUDA(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r) {
  int N = bboxes.size(0);
  int P = points.size(1);
  float cell_size = r;
  GridParams* h_params = new GridParams[N];
  int max_grid_total = 0;
  for (int i = 0; i < N; ++i) {
    SetupGridParams(
      bboxes.contiguous().data_ptr<float>() + i*6,
      cell_size,
      &h_params[i]
    );
    max_grid_total = std::max(max_grid_total, h_params[i].grid_total);
  }

  GridParams* d_params;
  cudaMalloc((void**)&d_params, N*sizeof(GridParams));
  cudaMemcpy(d_params, h_params, N*sizeof(GridParams), cudaMemcpyHostToDevice);

  auto int_dtype = lengths.options().dtype(at::kInt);

  auto grid_cnt = at::zeros({N, max_grid_total}, int_dtype);
  auto grid_cell = at::full({N, P}, -1, int_dtype); 
  auto grid_idx = at::full({N, P}, -1, int_dtype);

  InsertPointsCUDA(
    points,
    lengths,
    grid_cnt,
    grid_cell,
    grid_idx,
    max_grid_total,
    d_params
  );

  delete[] h_params;
  cudaFree(d_params);
  return std::make_tuple(grid_cnt, grid_cell, grid_idx);
}

template<int K>
__global__ void FindNbrsKernel(
    const float* __restrict__ points1,       
    const float* __restrict__ points2,       
    const long* __restrict__ lengths1,        
    const long* __restrict__ lengths2,
    const int* __restrict__ grid_off,
    const int* __restrict__ sorted_point_idx,
    float* __restrict__ dists,               
    long* __restrict__ idxs,                  
    int N,
    int P1,
    int P2,
    int G,
    const GridParams* params,                   // (N,)
    float r) {
  float min_dists[K];
  int min_idxs[K];
  float3 diff;
  float sqdist;
  float r2 = r*r;
  
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    if (p1 >= lengths1[n])
      continue;
    float3 cur_point;
    cur_point.x = points1[n*P1*3 + p1*3];
    cur_point.y = points1[n*P1*3 + p1*3 + 1];
    cur_point.z = points1[n*P1*3 + p1*3 + 2];
    int3 res = params[n].grid_res;
    float3 grid_min = params[n].grid_min;
    float grid_delta = params[n].grid_delta;

    int3  min_gc, max_gc;
    // gc.x = (int) ((cur_point.x-grid_min.x) * grid_delta);
    // gc.y = (int) ((cur_point.y-grid_min.y) * grid_delta);
    // gc.z = (int) ((cur_point.z-grid_min.z) * grid_delta);
    min_gc.x = (int) std::floor((cur_point.x-grid_min.x-r) * grid_delta);
    min_gc.y = (int) std::floor((cur_point.y-grid_min.y-r) * grid_delta);
    min_gc.z = (int) std::floor((cur_point.z-grid_min.z-r) * grid_delta);
    max_gc.x = (int) std::floor((cur_point.x-grid_min.x+r) * grid_delta);
    max_gc.y = (int) std::floor((cur_point.y-grid_min.y+r) * grid_delta);
    max_gc.z = (int) std::floor((cur_point.z-grid_min.z+r) * grid_delta);
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x=std::max(min_gc.x, 0); x<=std::min(max_gc.x, res.x-1); ++x) {
      for (int y=std::max(min_gc.y, 0); y<=std::min(max_gc.y, res.y-1); ++y) {
        for (int z=std::max(min_gc.z, 0); z<=std::min(max_gc.z, res.z-1); ++z) {
          int cell_idx = (x*res.y + y)*res.z + z;
          int p2_start = grid_off[n*G + cell_idx];
          int p2_end;
          if (cell_idx+1 == params[n].grid_total) {
            p2_end = lengths2[n];
          }
          else {
            p2_end = grid_off[n*G+cell_idx+1]; 
          }
          for (int p2=p2_start; p2<p2_end; ++p2) {
            diff.x = points2[n*P2*3 + p2*3] - cur_point.x;
            diff.y = points2[n*P2*3 + p2*3 + 1] - cur_point.y;
            diff.z = points2[n*P2*3 + p2*3 + 2] - cur_point.z;
            sqdist = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
            if (sqdist <= r2) {
              mink.add(sqdist, sorted_point_idx[n*P2+p2]);
            }
          }
        }
      }
    }
    mink.sort();
    for (int k=0; k < mink.size(); ++k) {
      idxs[n*P1*K + p1*K + k] = min_idxs[k];
      dists[n*P1*K + p1*K + k] = min_dists[k];
    }
  }
}

template<int K>
struct FindNbrsKernelFunctor {
  static void run(
      size_t blocks,
      size_t threads,
      const float* __restrict__ points1,          // (N, P1, 3)
      const float* __restrict__ points2,          // (N, P2, 3)
      const long* __restrict__ lengths1,           // (N,)
      const long* __restrict__ lengths2,           // (N,)
      const int* __restrict__ grid_off,           // (N, G)
      const int* __restrict__ sorted_point_idx,   // (N, P)
      float* __restrict__ dists,                  // (N, P1, K)
      long* __restrict__ idxs,                     // (N, P1, K)
      int N,
      int P1,
      int P2,
      int G,
      const GridParams* params,                   // (N,)
      float r) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FindNbrsKernel<K><<<blocks, threads, 0, stream>>>(
      points1, points2, lengths1, lengths2, grid_off, sorted_point_idx,
      dists, idxs, N, P1, P2, G, params, r);
  }
};

constexpr int MIN_K = 1;
constexpr int MAX_K = 32;

std::tuple<at::Tensor, at::Tensor> FindNbrsCUDA(
    const at::Tensor points1,
    const at::Tensor points2,
    const at::Tensor lengths1,
    const at::Tensor lengths2,
    const at::Tensor grid_off,
    const at::Tensor sorted_point_idx,
    const GridParams* params,
    int K,
    float r) {
  at::TensorArg points1_t{points1, "points1", 1};
  at::TensorArg points2_t{points2, "points2", 2};
  at::TensorArg lengths1_t{lengths1, "lengths1", 3};
  at::TensorArg lengths2_t{lengths2, "lengths2", 4};
  at::TensorArg grid_off_t{grid_off, "grid_off", 5};
  at::TensorArg sorted_point_idx_t{sorted_point_idx, "sorted_point_idx", 6};

  at::CheckedFrom c = "FindNbrsCUDA";
  at::checkAllSameGPU(c, {points1_t, points2_t, lengths1_t, lengths2_t, grid_off_t, sorted_point_idx_t});
  at::checkAllSameType(c, {points1_t, points2_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  at::checkAllSameType(c, {grid_off_t, sorted_point_idx_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int N = points1.size(0);
  int P1 = points1.size(1);
  int P2 = points2.size(1);
  int G = grid_off.size(1);
  
  auto idxs = at::full({N, P1, K}, -1, lengths1.options());
  auto dists = at::full({N, P1, K}, -1, points1.options());

  int threads = 256;
  int blocks = 256;

  DispatchKernel1D<FindNbrsKernelFunctor, MIN_K, MAX_K>( 
    K,
    blocks,
    threads,
    points1.contiguous().data_ptr<float>(),
    points2.contiguous().data_ptr<float>(),
    lengths1.contiguous().data_ptr<long>(),
    lengths2.contiguous().data_ptr<long>(),
    grid_off.contiguous().data_ptr<int>(),
    sorted_point_idx.contiguous().data_ptr<int>(),
    dists.data_ptr<float>(),
    idxs.data_ptr<long>(),
    N,
    P1,
    P2,
    G,
    params,
    r
  );

  /*
  // TODO: correctly use DispatchKernel1D here
  FindNbrsKernel<5><<<blocks, threads, 0, stream>>>(
    points1.contiguous().data_ptr<float>(),
    points2.contiguous().data_ptr<float>(),
    lengths1.contiguous().data_ptr<long>(),
    lengths2.contiguous().data_ptr<long>(),
    grid_off.contiguous().data_ptr<int>(),
    sorted_point_idx.contiguous().data_ptr<int>(),
    dists.data_ptr<float>(),
    idxs.data_ptr<long>(),
    N,
    P1,
    P2,
    G,
    params,
    r
  );
  */
  return std::make_tuple(idxs, dists);
}

std::tuple<at::Tensor, at::Tensor> TestFindNbrsCUDA(
    const at::Tensor bboxes,  
    const at::Tensor points1,  
    const at::Tensor points2,
    const at::Tensor lengths1,
    const at::Tensor lengths2,
    int K,
    float r) {
  int N = points1.size(0);
  int P1 = points1.size(1);
  int P2 = points2.size(1);
  float cell_size = r;
  GridParams* h_params = new GridParams[N];
  int max_grid_total = 0;
  for (int i = 0; i < N; ++i) {
    SetupGridParams(
      bboxes.contiguous().data_ptr<float>() + i*6,
      cell_size,
      &h_params[i]
    );
    max_grid_total = std::max(max_grid_total, h_params[i].grid_total);
  }

  GridParams* d_params;
  cudaMalloc((void**)&d_params, N*sizeof(GridParams));
  cudaMemcpy(d_params, h_params, N*sizeof(GridParams), cudaMemcpyHostToDevice);

  auto int_dtype = lengths2.options().dtype(at::kInt);

  auto grid_cnt = at::zeros({N, max_grid_total}, int_dtype);
  auto grid_cell = at::full({N, P2}, -1, int_dtype); 
  auto grid_idx = at::full({N, P2}, -1, int_dtype);

  InsertPointsCUDA(
    points2,
    lengths2,
    grid_cnt,
    grid_cell,
    grid_idx,
    max_grid_total,
    d_params
  );

  auto grid_off = PrefixSumCUDA(grid_cnt, h_params);

  auto sorted_points2 = at::zeros({N, P2, 3}, points2.options());
  auto sorted_point_idx = at::full({N, P2}, -1, int_dtype);

  CountingSortCUDA(
    points2,
    lengths2,
    grid_cell,
    grid_idx,
    grid_off,
    sorted_points2,
    sorted_point_idx
  );

  auto results = FindNbrsCUDA(
    points1,
    sorted_points2,
    lengths1,
    lengths2,
    grid_off,
    sorted_point_idx,
    d_params,
    K,
    r
  );

  delete[] h_params;
  cudaFree(d_params);
  return results;
}
