#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "utils/dispatch.cuh"
#include "utils/mink.cuh"

template <typename scalar_t, int64_t D, int64_t K>
__global__ void FRNNBruteForceKernel(const scalar_t *__restrict__ points1,
                                     const scalar_t *__restrict__ points2,
                                     const int64_t *__restrict__ lengths1,
                                     const int64_t *__restrict__ lengths2,
                                     scalar_t *__restrict__ dists,
                                     int64_t *__restrict__ idxs, int N, int P1,
                                     int P2, float r2) {
  scalar_t cur_point[D];
  scalar_t min_dists[K];
  int min_idxs[K];
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    if (p1 >= lengths1[n]) continue;
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    int length2 = lengths2[n];
    MinK<scalar_t, int> mink(min_dists, min_idxs, K);
    for (int p2 = 0; p2 < length2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        dist += diff * diff;
      }
      if (dist >= r2) continue;
      mink.add(dist, p2);
    }
    mink.sort();
    for (int k = 0; k < mink.size(); ++k) {
      // if (min_dists[k] >= r2)
      //   break;
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + p1 * K + k] = min_dists[k];
    }
  }
}

// This is a shim so we can dispatch using DispatchKernel2D
template <typename scalar_t, int64_t D, int64_t K>
struct FRNNBruteForceFunctor {
  static void run(int blocks, int threads, const scalar_t *__restrict__ points1,
                  const scalar_t *__restrict__ points2,
                  const int64_t *__restrict__ lengths1,
                  const int64_t *__restrict__ lengths2,
                  scalar_t *__restrict__ dists, int64_t *__restrict__ idxs,
                  int N, int P1, int P2, float r2) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FRNNBruteForceKernel<scalar_t, D, K><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, dists, idxs, N, P1, P2, r2);
  }
};

constexpr int V2_MIN_D = 1;
constexpr int V2_MAX_D = 8;
constexpr int V2_MIN_K = 1;
constexpr int V2_MAX_K = 64;

std::tuple<at::Tensor, at::Tensor> FRNNBruteForceCUDA(
    const at::Tensor &p1, const at::Tensor &p2, const at::Tensor &lengths1,
    const at::Tensor &lengths2, int K, float r) {
  // Check inputs are on the same device
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "FRNNBruteForceCUDA";
  at::checkAllSameGPU(c, {p1_t, p2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto N = p1.size(0);
  auto P1 = p1.size(1);
  auto P2 = p2.size(1);
  auto D = p2.size(2);
  int64_t K_64 = K;
  float r2 = r * r;

  TORCH_CHECK(p2.size(2) == D, "Point sets must have the same last dimension");
  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::full({N, P1, K}, -1, long_dtype);
  auto dists = at::full({N, P1, K}, -1, p1.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, dists);
  }

  AT_ASSERTM(D >= V2_MIN_D && D <= V2_MAX_D && K >= V2_MIN_K && D <= V2_MAX_K,
             "Invalid range for K or D");

  int threads = 256;
  int blocks = 256;
  AT_DISPATCH_FLOATING_TYPES(
      p1.scalar_type(), "frnn_kernel_cuda", ([&] {
        DispatchKernel2D<FRNNBruteForceFunctor, scalar_t, V2_MIN_D, V2_MAX_D,
                         V2_MIN_K, V2_MAX_K>(
            D, K_64, blocks, threads, p1.contiguous().data_ptr<scalar_t>(),
            p2.contiguous().data_ptr<scalar_t>(),
            lengths1.contiguous().data_ptr<int64_t>(),
            lengths2.contiguous().data_ptr<int64_t>(),
            dists.data_ptr<scalar_t>(), idxs.data_ptr<int64_t>(), N, P1, P2,
            r2);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(idxs, dists);
}