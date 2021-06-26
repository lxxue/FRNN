#include "grid/insert_points.h"

__global__ void InsertPoints2DKernel(const float *__restrict__ points,
                                     const long *__restrict__ lengths,
                                     const float *__restrict__ params,
                                     int *__restrict__ grid_cnt,
                                     int *__restrict__ grid_cell,
                                     int *__restrict__ grid_idx, int N, int P,
                                     int G) {
  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n]) continue;

    float grid_min_x = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_MIN_X];
    float grid_min_y = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_MIN_Y];
    float grid_delta = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_DELTA];
    int grid_res_x = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_RES_X];
    int grid_res_y = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_RES_Y];

    int gc_x = (int)((points[(n * P + p) * 2 + 0] - grid_min_x) * grid_delta);
    int gc_y = (int)((points[(n * P + p) * 2 + 1] - grid_min_y) * grid_delta);

    gc_x = max(min(gc_x, grid_res_x - 1), 0);
    gc_y = max(min(gc_y, grid_res_y - 1), 0);

    int gs = gc_x * grid_res_y + gc_y;
    grid_cell[n * P + p] = gs;
    grid_idx[n * P + p] = atomicAdd(&grid_cnt[n * G + gs], 1);
  }
}

/*
__global__ void InsertPoints3DKernel(const float *__restrict__ points,
                                     const long *__restrict__ lengths,
                                     const float *__restrict__ params,
                                     int *__restrict__ grid_cnt,
                                     int *__restrict__ grid_cell,
                                     int *__restrict__ grid_idx, int N, int P,
                                     int G) {
  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n]) continue;

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];

    int gc_x = (int)((points[(n * P + p) * 3 + 0] - grid_min_x) * grid_delta);
    int gc_y = (int)((points[(n * P + p) * 3 + 1] - grid_min_y) * grid_delta);
    int gc_z = (int)((points[(n * P + p) * 3 + 2] - grid_min_z) * grid_delta);

    gc_x = max(min(gc_x, grid_res_x - 1), 0);
    gc_y = max(min(gc_y, grid_res_y - 1), 0);
    gc_z = max(min(gc_z, grid_res_z - 1), 0);

    int gs = (gc_x * grid_res_y + gc_y) * grid_res_z + gc_z;
    grid_cell[n * P + p] = gs;
    grid_idx[n * P + p] = atomicAdd(&grid_cnt[n * G + gs], 1);
  }
}
*/

template <int D>
__global__ void InsertPointsNDKernel(const float *__restrict__ points,
                                     const long *__restrict__ lengths,
                                     const float *__restrict__ params,
                                     int *__restrict__ grid_cnt,
                                     int *__restrict__ grid_cell,
                                     int *__restrict__ grid_idx, int N, int P,
                                     int G) {
  int chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p = start_point + threadIdx.x;
    if (p >= lengths[n]) continue;

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];

    int gc_x = (int)((points[(n * P + p) * D + 0] - grid_min_x) * grid_delta);
    int gc_y = (int)((points[(n * P + p) * D + 1] - grid_min_y) * grid_delta);
    int gc_z = (int)((points[(n * P + p) * D + 2] - grid_min_z) * grid_delta);

    gc_x = max(min(gc_x, grid_res_x - 1), 0);
    gc_y = max(min(gc_y, grid_res_y - 1), 0);
    gc_z = max(min(gc_z, grid_res_z - 1), 0);

    int gs = (gc_x * grid_res_y + gc_y) * grid_res_z + gc_z;
    grid_cell[n * P + p] = gs;
    grid_idx[n * P + p] = atomicAdd(&grid_cnt[n * G + gs], 1);
  }
}

template <int D>
struct InsertPointsNDKernelFunctor {
  static void run(int blocks, int threads, const float *__restrict__ points,
                  const long *__restrict__ lengths,
                  const float *__restrict__ params, int *__restrict__ grid_cnt,
                  int *__restrict__ grid_cell, int *__restrict__ grid_idx,
                  int N, int P, int G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    InsertPointsNDKernel<D><<<blocks, threads, 0, stream>>>(
        points, lengths, params, grid_cnt, grid_cell, grid_idx, N, P, G);
  }
};

void InsertPointsCUDA(const at::Tensor points,   // (N, P, D)
                      const at::Tensor lengths,  // (N,)
                      const at::Tensor params,   // (N, 6/8)
                      at::Tensor grid_cnt,       // (N, G)
                      at::Tensor grid_cell,      // (N, P)
                      at::Tensor grid_idx,       // (N, P)
                      int G) {
  at::TensorArg points_t{points, "points", 1};
  at::TensorArg lengths_t{lengths, "lengths", 2};
  at::TensorArg params_t{params, "params", 3};
  at::TensorArg grid_cnt_t{grid_cnt, "grid_cnt", 4};
  at::TensorArg grid_cell_t{grid_cell, "grid_cell", 5};
  at::TensorArg grid_idx_t{grid_idx, "grid_idx", 6};

  at::CheckedFrom c = "InsertPointsCUDA";
  at::checkAllSameGPU(
      c, {points_t, lengths_t, params_t, grid_cnt_t, grid_cell_t, grid_idx_t});
  at::checkAllSameType(c, {grid_cnt_t, grid_cell_t, grid_idx_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = 256;

  int D = points.size(2);
  if (D == 2) {
    InsertPoints2DKernel<<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<float>(),
        lengths.contiguous().data_ptr<long>(),
        params.contiguous().data_ptr<float>(),
        grid_cnt.contiguous().data_ptr<int>(),
        grid_cell.contiguous().data_ptr<int>(),
        grid_idx.contiguous().data_ptr<int>(), points.size(0), points.size(1),
        G);
    // } else if (D == 3) {
    //   InsertPoints3DKernel<<<blocks, threads, 0, stream>>>(
    //       points.contiguous().data_ptr<float>(),
    //       lengths.contiguous().data_ptr<long>(),
    //       params.contiguous().data_ptr<float>(),
    //       grid_cnt.contiguous().data_ptr<int>(),
    //       grid_cell.contiguous().data_ptr<int>(),
    //       grid_idx.contiguous().data_ptr<int>(), points.size(0),
    //       points.size(1), G);
  } else {
    DispatchKernel1D<InsertPointsNDKernelFunctor, V0_MIN_D, V0_MAX_D>(
        D, blocks, threads, points.contiguous().data_ptr<float>(),
        lengths.contiguous().data_ptr<long>(),
        params.contiguous().data_ptr<float>(),
        grid_cnt.contiguous().data_ptr<int>(),
        grid_cell.contiguous().data_ptr<int>(),
        grid_idx.contiguous().data_ptr<int>(), points.size(0), points.size(1),
        G);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}
