#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>

#include "grid.h"
#include "counting_sort.h"
#include "utils/mink.cuh"
// customized dispatch utils for our function type
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
  if (cell_size < res_min/GRID_3D_MAX_RES)
    cell_size = res_min / GRID_3D_MAX_RES;
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

__global__ void InsertPoints2DKernel(
    const float* __restrict__ points,
    const long* __restrict__ lengths,
    const float* __restrict__ params,
    // TODO: add __restrict__ flag here
    int* grid_cnt, // not sure if we can use __restrict__ here
    int* __restrict__ grid_cell,
    int* __restrict__ grid_idx,
    int N,
    int P,
    int G) {

  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk=blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n])
      continue;

    float grid_min_x = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_MIN_X];
    float grid_min_y = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_MIN_Y];
    float grid_delta = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_DELTA];
    int grid_res_x = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_RES_X];
    int grid_res_y = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_RES_Y];

    int gc_x = (int) ((points[(n*P+p)*2+0]-grid_min_x) * grid_delta);
    int gc_y = (int) ((points[(n*P+p)*2+1]-grid_min_y) * grid_delta);

    gc_x = max(min(gc_x, grid_res_x-1), 0);
    gc_y = max(min(gc_y, grid_res_y-1), 0);

    int gs = gc_x*grid_res_y + gc_y;
    grid_cell[n*P+p] = gs;
    grid_idx[n*P+p] = atomicAdd(&grid_cnt[n*G + gs], 1);
  } 
}

__global__ void InsertPoints3DKernel(
    const float* __restrict__ points,
    const long* __restrict__ lengths,
    const float* __restrict__ params,
    // TODO: add __restrict__ flag here
    int* grid_cnt, // not sure if we can use __restrict__ here
    int* __restrict__ grid_cell,
    int* __restrict__ grid_idx,
    int N,
    int P,
    int G) {

  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk=blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n])
      continue;

    float grid_min_x = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_MIN_X];
    float grid_min_y = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_MIN_Y];
    float grid_min_z = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_MIN_Z];
    float grid_delta = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_DELTA];
    int grid_res_x = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_RES_X];
    int grid_res_y = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_RES_Y];
    int grid_res_z = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_RES_Z];

    int gc_x = (int) ((points[(n*P+p)*3+0]-grid_min_x) * grid_delta);
    int gc_y = (int) ((points[(n*P+p)*3+1]-grid_min_y) * grid_delta);
    int gc_z = (int) ((points[(n*P+p)*3+2]-grid_min_z) * grid_delta);

    gc_x = max(min(gc_x, grid_res_x-1), 0);
    gc_y = max(min(gc_y, grid_res_y-1), 0);
    gc_z = max(min(gc_z, grid_res_z-1), 0);

    int gs = (gc_x*grid_res_y + gc_y) * grid_res_z + gc_z;
    grid_cell[n*P+p] = gs;
    grid_idx[n*P+p] = atomicAdd(&grid_cnt[n*G + gs], 1);
  } 
}

void InsertPointsCUDA(
    const at::Tensor points,    // (N, P, 2/3)
    const at::Tensor lengths,   // (N,)
    const at::Tensor params,    // (N, 6/8)
    at::Tensor grid_cnt,        // (N, G)
    at::Tensor grid_cell,       // (N, P)      
    at::Tensor grid_idx,        // (N, P)
    int G) {
  
  at::TensorArg points_t{points, "points", 1};
  at::TensorArg lengths_t{lengths, "lengths", 2};
  at::TensorArg params_t{params, "params", 3};
  at::TensorArg grid_cnt_t{grid_cnt, "grid_cnt", 4};
  at::TensorArg grid_cell_t{grid_cell, "grid_cell", 5};
  at::TensorArg grid_idx_t{grid_idx, "grid_idx", 6};

  at::CheckedFrom c = "InsertPointsCUDA";
  at::checkAllSameGPU(c, {points_t, lengths_t, params_t, grid_cnt_t, grid_cell_t, grid_idx_t});
  at::checkAllSameType(c, {grid_cnt_t, grid_cell_t, grid_idx_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = 256;

  const auto D = points.size(2);
  TORCH_CHECK(D == 2 || D == 3, "for now only 2D and 3D are supported");
  if (D == 2) {
    InsertPoints2DKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      lengths.contiguous().data_ptr<long>(),
      params.contiguous().data_ptr<float>(),
      grid_cnt.contiguous().data_ptr<int>(),
      grid_cell.contiguous().data_ptr<int>(),
      grid_idx.contiguous().data_ptr<int>(),
      points.size(0),
      points.size(1),
      G
    );
  }
  else {
    InsertPoints3DKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      lengths.contiguous().data_ptr<long>(),
      params.contiguous().data_ptr<float>(),
      grid_cnt.contiguous().data_ptr<int>(),
      grid_cell.contiguous().data_ptr<int>(),
      grid_idx.contiguous().data_ptr<int>(),
      points.size(0),
      points.size(1),
      G
    );
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

template<int K>
__global__ void FindNbrs2DKernel(
    const float* __restrict__ points1,       
    const float* __restrict__ points2,       
    const long* __restrict__ lengths1,        
    const long* __restrict__ lengths2,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points1_idxs,
    const int* __restrict__ sorted_points2_idxs,
    const float* __restrict__ params,
    float* __restrict__ dists,               
    long* __restrict__ idxs,                  
    int N,
    int P1,
    int P2,
    int G,
    const float* __restrict__ rs,
    const float* __restrict__ r2s) {
  float min_dists[K];
  int min_idxs[K];
  float2 diff;
  float sqdist;
  
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n*P1+p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float2 cur_point;
    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    cur_point.x = points1[n*P1*2 + p1*2];
    cur_point.y = points1[n*P1*2 + p1*2 + 1];

    float grid_min_x = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_MIN_X];
    float grid_min_y = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_MIN_Y];
    float grid_delta = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_DELTA];
    int grid_res_x = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_RES_X];
    int grid_res_y = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_RES_Y];
    int grid_total = params[n*GRID_2D_PARAMS_SIZE+GRID_2D_TOTAL];

    int min_gc_x = (int) std::floor((cur_point.x-grid_min_x-cur_r) * grid_delta);
    int min_gc_y = (int) std::floor((cur_point.y-grid_min_y-cur_r) * grid_delta);
    int max_gc_x = (int) std::floor((cur_point.x-grid_min_x+cur_r) * grid_delta);
    int max_gc_y = (int) std::floor((cur_point.y-grid_min_y+cur_r) * grid_delta);
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x=max(min_gc_x, 0); x<=min(max_gc_x, grid_res_x-1); ++x) {
      for (int y=max(min_gc_y, 0); y<=min(max_gc_y, grid_res_y-1); ++y) {
        int cell_idx = x*grid_res_y + y;
        int p2_start = pc2_grid_off[n*G + cell_idx];
        int p2_end;
        if (cell_idx+1 == grid_total) {
          p2_end = lengths2[n];
        }
        else {
          p2_end = pc2_grid_off[n*G+cell_idx+1]; 
        }
        for (int p2=p2_start; p2<p2_end; ++p2) {
          diff.x = points2[n*P2*2 + p2*2] - cur_point.x;
          diff.y = points2[n*P2*2 + p2*2 + 1] - cur_point.y;
          sqdist = diff.x*diff.x + diff.y*diff.y;
          if (sqdist <= cur_r2) {
            mink.add(sqdist, sorted_points2_idxs[n*P2+p2]);
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    for (int k=0; k < mink.size(); ++k) {
      idxs[n*P1*K + old_p1*K + k] = min_idxs[k];
      dists[n*P1*K + old_p1*K + k] = min_dists[k];
    }
  }
}


template<int K>
__global__ void FindNbrs3DKernel(
    const float* __restrict__ points1,       
    const float* __restrict__ points2,       
    const long* __restrict__ lengths1,        
    const long* __restrict__ lengths2,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points1_idxs,
    const int* __restrict__ sorted_points2_idxs,
    const float* __restrict__ params,
    float* __restrict__ dists,               
    long* __restrict__ idxs,                  
    int N,
    int P1,
    int P2,
    int G,
    const float* __restrict__ rs,
    const float* __restrict__ r2s) {
  float min_dists[K];
  int min_idxs[K];
  float3 diff;
  float sqdist;
  
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n*P1+p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float3 cur_point;
    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    cur_point.x = points1[n*P1*3 + p1*3];
    cur_point.y = points1[n*P1*3 + p1*3 + 1];
    cur_point.z = points1[n*P1*3 + p1*3 + 2];

    float grid_min_x = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_MIN_X];
    float grid_min_y = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_MIN_Y];
    float grid_min_z = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_MIN_Z];
    float grid_delta = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_DELTA];
    int grid_res_x = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_RES_X];
    int grid_res_y = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_RES_Y];
    int grid_res_z = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_RES_Z];
    int grid_total = params[n*GRID_3D_PARAMS_SIZE+GRID_3D_TOTAL];

    int min_gc_x = (int) std::floor((cur_point.x-grid_min_x-cur_r) * grid_delta);
    int min_gc_y = (int) std::floor((cur_point.y-grid_min_y-cur_r) * grid_delta);
    int min_gc_z = (int) std::floor((cur_point.z-grid_min_z-cur_r) * grid_delta);
    int max_gc_x = (int) std::floor((cur_point.x-grid_min_x+cur_r) * grid_delta);
    int max_gc_y = (int) std::floor((cur_point.y-grid_min_y+cur_r) * grid_delta);
    int max_gc_z = (int) std::floor((cur_point.z-grid_min_z+cur_r) * grid_delta);
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x=max(min_gc_x, 0); x<=min(max_gc_x, grid_res_x-1); ++x) {
      for (int y=max(min_gc_y, 0); y<=min(max_gc_y, grid_res_y-1); ++y) {
        for (int z=max(min_gc_z, 0); z<=min(max_gc_z, grid_res_z-1); ++z) {
          int cell_idx = (x*grid_res_y + y)*grid_res_z + z;
          int p2_start = pc2_grid_off[n*G + cell_idx];
          int p2_end;
          if (cell_idx+1 == grid_total) {
            p2_end = lengths2[n];
          }
          else {
            p2_end = pc2_grid_off[n*G+cell_idx+1]; 
          }
          for (int p2=p2_start; p2<p2_end; ++p2) {
            diff.x = points2[n*P2*3 + p2*3] - cur_point.x;
            diff.y = points2[n*P2*3 + p2*3 + 1] - cur_point.y;
            diff.z = points2[n*P2*3 + p2*3 + 2] - cur_point.z;
            sqdist = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
            if (sqdist <= cur_r2) {
              mink.add(sqdist, sorted_points2_idxs[n*P2+p2]);
            }
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    for (int k=0; k < mink.size(); ++k) {
      idxs[n*P1*K + old_p1*K + k] = min_idxs[k];
      dists[n*P1*K + old_p1*K + k] = min_dists[k];
    }
  }
}

template<int K>
struct FindNbrsKernelFunctor {
  static void run(
      int blocks,
      int threads,
      int D,
      const float* __restrict__ points1,              // (N, P1, 2/3)
      const float* __restrict__ points2,              // (N, P2, 2/3)
      const long* __restrict__ lengths1,              // (N,)
      const long* __restrict__ lengths2,              // (N,)
      const int* __restrict__ pc2_grid_off,           // (N, G)
      const int* __restrict__ sorted_points1_idxs,    // (N, P)
      const int* __restrict__ sorted_points2_idxs,    // (N, P)
      const float* __restrict__ params,               // (N,)
      float* __restrict__ dists,                      // (N, P1, K)
      long* __restrict__ idxs,                        // (N, P1, K)
      int N,
      int P1,
      int P2,
      int G,
      const float* rs,
      const float* r2s) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (D == 2) {
      FindNbrs2DKernel<K><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, pc2_grid_off, 
        sorted_points1_idxs, sorted_points2_idxs, params,
        dists, idxs, N, P1, P2, G, rs, r2s);
    }
    else {
      FindNbrs3DKernel<K><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, pc2_grid_off, 
        sorted_points1_idxs, sorted_points2_idxs, params,
        dists, idxs, N, P1, P2, G, rs, r2s);
    }
  }
};

// TODO: figure out max & min; sanity check in python
constexpr int MIN_K = 1;
constexpr int MAX_K = 32;

std::tuple<at::Tensor, at::Tensor> FindNbrsCUDA(
    const at::Tensor points1,
    const at::Tensor points2,
    const at::Tensor lengths1,
    const at::Tensor lengths2,
    const at::Tensor pc2_grid_off,
    const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs,
    const at::Tensor params,
    int K,
    const at::Tensor rs,
    const at::Tensor r2s) {

  at::TensorArg points1_t{points1, "points1", 1};
  at::TensorArg points2_t{points2, "points2", 2};
  at::TensorArg lengths1_t{lengths1, "lengths1", 3};
  at::TensorArg lengths2_t{lengths2, "lengths2", 4};
  at::TensorArg pc2_grid_off_t{pc2_grid_off, "pc2_grid_off", 5};
  at::TensorArg sorted_points1_idxs_t{sorted_points1_idxs, "sorted_points1_idxs", 6};
  at::TensorArg sorted_points2_idxs_t{sorted_points2_idxs, "sorted_points2_idxs", 7};
  at::TensorArg params_t{params, "params", 8};
  at::TensorArg rs_t{rs, "rs", 10};
  at::TensorArg r2s_t{r2s, "r2s", 11};

  at::CheckedFrom c = "FindNbrsCUDA";
  at::checkAllSameGPU(c, {points1_t, points2_t, lengths1_t, lengths2_t, 
    pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t, r2s_t});
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t, r2s_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  at::checkAllSameType(c, {pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int N = points1.size(0);
  int P1 = points1.size(1);
  int D = points1.size(2);
  int P2 = points2.size(1);
  int G = pc2_grid_off.size(1);
  
  auto idxs = at::full({N, P1, K}, -1, lengths1.options());
  auto dists = at::full({N, P1, K}, -1, points1.options());

  int threads = 256;
  int blocks = 256;

  DispatchKernel1D<FindNbrsKernelFunctor, MIN_K, MAX_K>( 
    K,
    blocks,
    threads,
    D,
    points1.contiguous().data_ptr<float>(),
    points2.contiguous().data_ptr<float>(),
    lengths1.contiguous().data_ptr<long>(),
    lengths2.contiguous().data_ptr<long>(),
    pc2_grid_off.contiguous().data_ptr<int>(),
    sorted_points1_idxs.contiguous().data_ptr<int>(),
    sorted_points2_idxs.contiguous().data_ptr<int>(),
    params.contiguous().data_ptr<float>(),
    dists.data_ptr<float>(),
    idxs.data_ptr<long>(),
    N,
    P1,
    P2,
    G,
    rs.data_ptr<float>(),
    r2s.data_ptr<float>()
  );

  return std::make_tuple(idxs, dists);
}