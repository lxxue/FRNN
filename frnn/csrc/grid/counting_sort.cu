#include "grid/counting_sort.h"

__global__ void CountingSort2DKernel(
    const float *__restrict__ points,   // (N, P, 2)
    const long *__restrict__ lengths,   // (N,)
    const int *__restrict__ grid_cell,  // (N, P)
    const int *__restrict__ grid_idx,   // (N, P)
    const int *__restrict__ grid_off,   // (N, G)
    float *__restrict__ sorted_points,  // (N, P, 2)
    // sorted[n, i] = unsorted[n, idxs[i]]
    int *__restrict__ sorted_points_idxs,  // (N, P)
    int N, int P, int G) {
  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n]) continue;

    int cell_idx = grid_cell[n * P + p];
    int idx = grid_idx[n * P + p];
    assert(cell_idx < G);
    int sorted_idx = grid_off[n * G + cell_idx] + idx;
    assert(sorted_idx >= 0 && sorted_idx < lengths[n]);

    sorted_points[n * P * 2 + sorted_idx * 2] = points[n * P * 2 + p * 2];
    sorted_points[n * P * 2 + sorted_idx * 2 + 1] =
        points[n * P * 2 + p * 2 + 1];

    sorted_points_idxs[n * P + sorted_idx] = p;
  }
}

/*
__global__ void CountingSort3DKernel(
    const float *__restrict__ points,      // (N, P, 3)
    const long *__restrict__ lengths,      // (N,)
    const int *__restrict__ grid_cell,     // (N, P)
    const int *__restrict__ grid_idx,      // (N, P)
    const int *__restrict__ grid_off,      // (N, G)
    float *__restrict__ sorted_points,     // (N, P, 3)
    int *__restrict__ sorted_points_idxs,  // (N, P): sorted[n, i] = unsorted[n,
                                           // idxs[i]]
    int N, int P, int G) {
  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n]) continue;

    int cell_idx = grid_cell[n * P + p];
    int idx = grid_idx[n * P + p];
    assert(cell_idx < G);
    int sorted_idx = grid_off[n * G + cell_idx] + idx;
    assert(sorted_idx >= 0 && sorted_idx < lengths[n]);

    sorted_points[n * P * 3 + sorted_idx * 3] = points[n * P * 3 + p * 3];
    sorted_points[n * P * 3 + sorted_idx * 3 + 1] =
        points[n * P * 3 + p * 3 + 1];
    sorted_points[n * P * 3 + sorted_idx * 3 + 2] =
        points[n * P * 3 + p * 3 + 2];

    sorted_points_idxs[n * P + sorted_idx] = p;
  }
}
*/

template <int D>
__global__ void CountingSortNDKernel(
    const float *__restrict__ points,      // (N, P, 3)
    const long *__restrict__ lengths,      // (N,)
    const int *__restrict__ grid_cell,     // (N, P)
    const int *__restrict__ grid_idx,      // (N, P)
    const int *__restrict__ grid_off,      // (N, G)
    float *__restrict__ sorted_points,     // (N, P, 3)
    int *__restrict__ sorted_points_idxs,  // (N, P):
                                           // sorted[n, i] = unsorted[n,idxs[i]]
    int N, int P, int G) {
  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n]) continue;

    int cell_idx = grid_cell[n * P + p];
    int idx = grid_idx[n * P + p];
    assert(cell_idx < G);
    int sorted_idx = grid_off[n * G + cell_idx] + idx;
    assert(sorted_idx >= 0 && sorted_idx < lengths[n]);

    for (int d = 0; d < D; ++d) {
      sorted_points[n * P * D + sorted_idx * D + d] =
          points[n * P * D + p * D + d];
    }

    sorted_points_idxs[n * P + sorted_idx] = p;
  }
}

template <int D>
struct CountingSortNDKernelFunctor {
  static void run(int blocks, int threads, const float *__restrict__ points,
                  const long *__restrict__ lengths,
                  const int *__restrict__ grid_cell,
                  const int *__restrict__ grid_idx,
                  const int *__restrict__ grid_off,
                  float *__restrict__ sorted_points,
                  int *__restrict__ sorted_points_idxs, int N, int P, int G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CountingSortNDKernel<D><<<blocks, threads, 0, stream>>>(
        points, lengths, grid_cell, grid_idx, grid_off, sorted_points,
        sorted_points_idxs, N, P, G);
  }
};

void CountingSortCUDA(const at::Tensor points, const at::Tensor lengths,
                      const at::Tensor grid_cell, const at::Tensor grid_idx,
                      const at::Tensor grid_off, at::Tensor sorted_points,
                      at::Tensor sorted_points_idxs) {
  at::TensorArg points_t{points, "points", 1};
  at::TensorArg lengths_t{lengths, "lengths", 2};
  at::TensorArg grid_cell_t{grid_cell, "grid_cell", 3};
  at::TensorArg grid_idx_t{grid_idx, "grid_idx", 4};
  at::TensorArg grid_off_t{grid_off, "grid_off", 5};
  at::TensorArg sorted_points_t{sorted_points, "sorted_points", 6};
  at::TensorArg sorted_points_idxs_t{sorted_points_idxs, "sorted_points_idxs",
                                     7};

  at::CheckedFrom c = "CountingSortCUDA";
  at::checkAllSameGPU(c, {points_t, lengths_t, grid_cell_t, grid_idx_t,
                          grid_off_t, sorted_points_t, sorted_points_idxs_t});
  at::checkAllSameType(
      c, {grid_cell_t, grid_idx_t, grid_off_t, sorted_points_idxs_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = 256;
  int N = points.size(0);
  int P = points.size(1);
  int D = points.size(2);
  // assert(D == 2 || D == 3);
  int G = grid_off.size(1);

  if (D == 2) {
    CountingSort2DKernel<<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<float>(),
        lengths.contiguous().data_ptr<long>(),
        grid_cell.contiguous().data_ptr<int>(),
        grid_idx.contiguous().data_ptr<int>(),
        grid_off.contiguous().data_ptr<int>(),
        sorted_points.contiguous().data_ptr<float>(),
        sorted_points_idxs.contiguous().data_ptr<int>(), N, P, G);
  } else {
    // CountingSort3DKernel<<<blocks, threads, 0, stream>>>(
    //     points.contiguous().data_ptr<float>(),
    //     lengths.contiguous().data_ptr<long>(),
    //     grid_cell.contiguous().data_ptr<int>(),
    //     grid_idx.contiguous().data_ptr<int>(),
    //     grid_off.contiguous().data_ptr<int>(),
    //     sorted_points.contiguous().data_ptr<float>(),
    //     sorted_points_idxs.contiguous().data_ptr<int>(), N, P, G);

    DispatchKernel1D<CountingSortNDKernelFunctor, V0_MIN_D, V0_MAX_D>(
        D, blocks, threads, points.contiguous().data_ptr<float>(),
        lengths.contiguous().data_ptr<long>(),
        grid_cell.contiguous().data_ptr<int>(),
        grid_idx.contiguous().data_ptr<int>(),
        grid_off.contiguous().data_ptr<int>(),
        sorted_points.contiguous().data_ptr<float>(),
        sorted_points_idxs.contiguous().data_ptr<int>(), N, P, G);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}