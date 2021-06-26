#include "backward/backward.h"

__global__ void FRNNBackward2DKernel(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const long *__restrict__ idxs, const float *__restrict__ grad_dists,
    float *__restrict__ grad_points1, float *__restrict__ grad_points2, int N,
    int P1, int P2, int K) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < N * P1 * K * 2; i += stride) {
    const int n = i / (P1 * K * 2);
    int rem = i % (P1 * K * 2);
    const int p1_idx = rem / (K * 2);
    rem = rem % (K * 2);
    const int k = rem / 2;
    const int d = rem % 2;

    const long num1 = lengths1[n];
    const long num2 = lengths2[n];
    if ((p1_idx < num1) && (k < num2)) {
      const long p2_idx = idxs[n * P1 * K + p1_idx * K + k];
      if (p2_idx < 0)
        // sentinel value -1 indicating no fixed radius negihbors here
        continue;
      const float grad_dist = grad_dists[n * P1 * K + p1_idx * K + k];

      const float diff = 2.0f * grad_dist *
                         (points1[n * P1 * 2 + p1_idx * 2 + d] -
                          points2[n * P2 * 2 + p2_idx * 2 + d]);
      atomicAdd(grad_points1 + n * P1 * 2 + p1_idx * 2 + d, diff);
      atomicAdd(grad_points2 + n * P2 * 2 + p2_idx * 2 + d, -1.0f * diff);
    }
  }
}

/*
__global__ void FRNNBackward3DKernel(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const long *__restrict__ idxs, const float *__restrict__ grad_dists,
    float *__restrict__ grad_points1, float *__restrict__ grad_points2, int N,
    int P1, int P2, int K) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < N * P1 * K * 3; i += stride) {
    const int n = i / (P1 * K * 3);
    int rem = i % (P1 * K * 3);
    const int p1_idx = rem / (K * 3);
    rem = rem % (K * 3);
    const int k = rem / 3;
    const int d = rem % 3;

    const long num1 = lengths1[n];
    const long num2 = lengths2[n];
    if ((p1_idx < num1) && (k < num2)) {
      const long p2_idx = idxs[n * P1 * K + p1_idx * K + k];
      if (p2_idx < 0)
        // sentinel value -1 indicating no fixed radius negihbors here
        continue;
      const float grad_dist = grad_dists[n * P1 * K + p1_idx * K + k];

      const float diff = 2.0f * grad_dist *
                         (points1[n * P1 * 3 + p1_idx * 3 + d] -
                          points2[n * P2 * 3 + p2_idx * 3 + d]);
      atomicAdd(grad_points1 + n * P1 * 3 + p1_idx * 3 + d, diff);
      atomicAdd(grad_points2 + n * P2 * 3 + p2_idx * 3 + d, -1.0f * diff);
    }
  }
}
*/
template <int D>
__global__ void FRNNBackwardNDKernel(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const long *__restrict__ idxs, const float *__restrict__ grad_dists,
    float *__restrict__ grad_points1, float *__restrict__ grad_points2, int N,
    int P1, int P2, int K) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < N * P1 * K * D; i += stride) {
    const int n = i / (P1 * K * D);
    int rem = i % (P1 * K * D);
    const int p1_idx = rem / (K * D);
    rem = rem % (K * D);
    const int k = rem / D;
    const int d = rem % D;

    const long num1 = lengths1[n];
    const long num2 = lengths2[n];
    if ((p1_idx < num1) && (k < num2)) {
      const long p2_idx = idxs[n * P1 * K + p1_idx * K + k];
      if (p2_idx < 0)
        // sentinel value -1 indicating no fixed radius negihbors here
        continue;
      const float grad_dist = grad_dists[n * P1 * K + p1_idx * K + k];

      const float diff = 2.0f * grad_dist *
                         (points1[n * P1 * D + p1_idx * D + d] -
                          points2[n * P2 * D + p2_idx * D + d]);
      atomicAdd(grad_points1 + n * P1 * D + p1_idx * D + d, diff);
      atomicAdd(grad_points2 + n * P2 * D + p2_idx * D + d, -1.0f * diff);
    }
  }
}

template <int D>
struct FRNNBackwardNDKernelFunctor {
  static void run(
      int blocks, int threads, const float *__restrict__ points1,
      const float *__restrict__ points2, const long *__restrict__ lengths1,
      const long *__restrict__ lengths2, const long *__restrict__ idxs,
      const float *__restrict__ grad_dists, float *__restrict__ grad_points1,
      float *__restrict__ grad_points2, int N, int P1, int P2, int K) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FRNNBackwardNDKernel<D><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, idxs, grad_dists, grad_points1,
        grad_points2, N, P1, P2, K);
  }
};

std::tuple<at::Tensor, at::Tensor> FRNNBackwardCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2, const at::Tensor idxs,
    const at::Tensor grad_dists) {
  at::TensorArg points1_t{points1, "points1", 1},
      points2_t{points2, "points2", 2}, lengths1_t{lengths1, "lenghts1", 3},
      lengths2_t{lengths2, "lengths2", 4}, idxs_t{idxs, "idxs", 5},
      grad_dists_t{grad_dists, "grad_dists", 6};
  at::CheckedFrom c = "FRNNBackwardCUDA";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, idxs_t, grad_dists_t});
  at::checkAllSameType(c, {points1_t, points2_t, grad_dists_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t, idxs_t});

  const int N = points1.size(0);
  const int P1 = points1.size(1);
  const int D = points1.size(2);
  const int P2 = points2.size(1);
  const int K = idxs.size(2);

  TORCH_CHECK(points1.size(2) == points2.size(2),
              "Two point clouds of different dimensions");
  // TORCH_CHECK(points1.size(2) == 2 || points1.size(2) == 3,
  //             "Only 2D/3D points are supported");
  TORCH_CHECK(idxs.size(0) == N,
              "FRNN idxs must have the same batch dimension");
  TORCH_CHECK(idxs.size(1) == P1,
              "FRNN idxs must have the same point dimension as P1");
  TORCH_CHECK(grad_dists.size(0) == N);
  TORCH_CHECK(grad_dists.size(1) == P1);
  TORCH_CHECK(grad_dists.size(2) == K);

  at::Tensor grad_points1 = at::zeros({N, P1, D}, points1.options());
  at::Tensor grad_points2 = at::zeros({N, P2, D}, points2.options());

  if (grad_points1.numel() == 0 || grad_points2.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_points1, grad_points2);
  }

  const int blocks = 64;
  const int threads = 512;

  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (D == 2) {
    FRNNBackward2DKernel<<<blocks, threads, 0, stream>>>(
        points1.contiguous().data_ptr<float>(),
        points2.contiguous().data_ptr<float>(),
        lengths1.contiguous().data_ptr<long>(),
        lengths2.contiguous().data_ptr<long>(),
        idxs.contiguous().data_ptr<long>(),
        grad_dists.contiguous().data_ptr<float>(),
        grad_points1.data_ptr<float>(), grad_points2.data_ptr<float>(), N, P1,
        P2, K);
  } else {
    // FRNNBackward3DKernel<<<blocks, threads, 0, stream>>>(
    //     points1.contiguous().data_ptr<float>(),
    //     points2.contiguous().data_ptr<float>(),
    //     lengths1.contiguous().data_ptr<long>(),
    //     lengths2.contiguous().data_ptr<long>(),
    //     idxs.contiguous().data_ptr<long>(),
    //     grad_dists.contiguous().data_ptr<float>(),
    //     grad_points1.data_ptr<float>(), grad_points2.data_ptr<float>(), N,
    //     P1, P2, K);

    DispatchKernel1D<FRNNBackwardNDKernelFunctor, V0_MIN_D, V0_MAX_D>(
        D, blocks, threads, points1.contiguous().data_ptr<float>(),
        points2.contiguous().data_ptr<float>(),
        lengths1.contiguous().data_ptr<long>(),
        lengths2.contiguous().data_ptr<long>(),
        idxs.contiguous().data_ptr<long>(),
        grad_dists.contiguous().data_ptr<float>(),
        grad_points1.data_ptr<float>(), grad_points2.data_ptr<float>(), N, P1,
        P2, K);
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_points1, grad_points2);
}