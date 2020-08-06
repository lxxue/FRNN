#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename idx_t>
__global__ void CountingSortKernel (
    const float* __restrict__ points,     // (N, P, 3)
    const idx_t* __restrict__ lengths,    // (N,)
    const idx_t* __restrict__ grid_cell,  // (N, P)
    const idx_t* __restrict__ grid_idx,   // (N, P)
    const idx_t* __restrict__ grid_off,   // (N, G)
    float* __restrict__ sorted_points,    // (N, P, 3)
    idx_t* __restrict__ sorted_grid_cell, // (N, P)
    idx_t* __restrict__ sorted_point_idx, // (N, P): new idx -> old idx
    size_t N,
    size_t P,
    size_t G) {
  const idx_t chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  const idx_t chunks_to_do = N * chunks_per_cloud;
  for (idx_t chunk=blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const idx_t n = chunk / chunks_per_cloud;
    const idx_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    idx_t p = start_point + threadIdx.x;
    if (p >= lengths[n])
      continue;

    idx_t cell_idx = grid_cell[n*P + p];
    idx_t idx = grid_idx[n*P + p];
    idx_t sorted_idx = grid_off[n*G + cell_idx] + idx;

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
  at::checkAllSameType(c, {lengths_t, grid_cell_t, grid_idx_t, grid_off_t,
      sorted_grid_cell_t, sorted_point_idx_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  int threads = 256;
  int blocks = 256;
  size_t N = points.size(0);
  size_t P = points.size(1);
  size_t G = grid_off.size(1);
  
  CountingSortKernel<int><<<blocks, threads, 0, stream>>>(
    points.contiguous().data_ptr<float>(),
    lengths.contiguous().data_ptr<int>(),
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