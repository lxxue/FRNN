#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void CountingSortKernel (
    const float* __restrict__ points,     // (N, P, 3)
    const long* __restrict__ lengths,    // (N,)
    const int* __restrict__ grid_cell,  // (N, P)
    const int* __restrict__ grid_idx,   // (N, P)
    const int* __restrict__ grid_off,   // (N, G)
    float* __restrict__ sorted_points,    // (N, P, 3)
    int* __restrict__ sorted_grid_cell, // (N, P)
    int* __restrict__ sorted_point_idx, // (N, P): new idx -> old idx
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

    int cell_idx = grid_cell[n*P + p];
    int idx = grid_idx[n*P + p];
    int sorted_idx = grid_off[n*G + cell_idx] + idx;

    sorted_points[n*P*3 + sorted_idx*3] = points[n*P*3 + p*3];
    sorted_points[n*P*3 + sorted_idx*3+1] = points[n*P*3 + p*3+1];
    sorted_points[n*P*3 + sorted_idx*3+2] = points[n*P*3 + p*3+2];

    sorted_grid_cell[n*P+sorted_idx] = cell_idx;
    sorted_point_idx[n*P+sorted_idx] = p;
  }
}

void CountingSortCUDA(
    const at::Tensor points,
    const at::Tensor lengths,
    const at::Tensor grid_cell,
    const at::Tensor grid_idx,
    const at::Tensor grid_off,
    at::Tensor sorted_points,
    at::Tensor sorted_grid_cell,
    at::Tensor sorted_point_idx) {

  at::TensorArg points_t{points, "points", 1};
  at::TensorArg lengths_t{lengths, "lengths", 2};
  at::TensorArg grid_cell_t{grid_cell, "grid_cell", 3};
  at::TensorArg grid_idx_t{grid_idx, "grid_idx", 4};
  at::TensorArg grid_off_t{grid_off, "grid_off", 5};
  at::TensorArg sorted_points_t{sorted_points, "sorted_points", 6};
  at::TensorArg sorted_grid_cell_t{sorted_grid_cell, "sorted_grid_cell", 7};
  at::TensorArg sorted_point_idx_t{sorted_point_idx, "sorted_point_idx", 8};
  
  at::CheckedFrom c = "CountingSortCUDA";
  at::checkAllSameGPU(c, {points_t, lengths_t, grid_cell_t, grid_idx_t, grid_off_t,
      sorted_points_t, sorted_grid_cell_t, sorted_point_idx_t});
  at::checkAllSameType(c, {grid_cell_t, grid_idx_t, grid_off_t,
      sorted_grid_cell_t, sorted_point_idx_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  int threads = 256;
  int blocks = 256;
  int N = points.size(0);
  int P = points.size(1);
  int G = grid_off.size(1);
  
  CountingSortKernel<<<blocks, threads, 0, stream>>>(
    points.contiguous().data_ptr<float>(),
    lengths.contiguous().data_ptr<long>(),
    grid_cell.contiguous().data_ptr<int>(),
    grid_idx.contiguous().data_ptr<int>(),
    grid_off.contiguous().data_ptr<int>(),
    sorted_points.contiguous().data_ptr<float>(),
    sorted_grid_cell.contiguous().data_ptr<int>(),
    sorted_point_idx.contiguous().data_ptr<int>(),
    N,
    P,
    G
  );
  AT_CUDA_CHECK(cudaGetLastError());
}