#include "grid/find_nbrs.h"

__global__ void FindNbrs2DKernelV1(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params, float *__restrict__ dists,
    long *__restrict__ idxs, int N, int P1, int P2, int G, int K,
    const float *__restrict__ rs, const float *__restrict__ r2s) {
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float2 cur_point;
    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    cur_point.x = points1[n * P1 * 2 + p1 * 2];
    cur_point.y = points1[n * P1 * 2 + p1 * 2 + 1];

    float grid_min_x = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_MIN_X];
    float grid_min_y = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_MIN_Y];
    float grid_delta = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_DELTA];
    int grid_res_x = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_RES_X];
    int grid_res_y = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_RES_Y];
    int grid_total = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point.x - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point.y - grid_min_y - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point.x - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point.y - grid_min_y + cur_r) * grid_delta);
    // use global memory directly
    int offset = n * P1 * K + old_p1 * K;
    MinK<float, long> mink(dists + offset, idxs + offset, K);
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        int cell_idx = x * grid_res_y + y;
        int p2_start = pc2_grid_off[n * G + cell_idx];
        int p2_end;
        if (cell_idx + 1 == grid_total) {
          p2_end = lengths2[n];
        } else {
          p2_end = pc2_grid_off[n * G + cell_idx + 1];
        }
        for (int p2 = p2_start; p2 < p2_end; ++p2) {
          float2 diff;
          diff.x = points2[n * P2 * 2 + p2 * 2] - cur_point.x;
          diff.y = points2[n * P2 * 2 + p2 * 2 + 1] - cur_point.y;
          float sqdist = diff.x * diff.x + diff.y * diff.y;
          if (sqdist <= cur_r2) {
            mink.add(sqdist, sorted_points2_idxs[n * P2 + p2]);
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    // for (int k = 0; k < mink.size(); ++k) {
    //   idxs[n * P1 * K + old_p1 * K + k] = min_idxs[k];
    //   dists[n * P1 * K + old_p1 * K + k] = min_dists[k];
    // }
  }
}

template <int K>
__global__ void FindNbrs2DKernelV2(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params, float *__restrict__ dists,
    long *__restrict__ idxs, int N, int P1, int P2, int G,
    const float *__restrict__ rs, const float *__restrict__ r2s) {
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
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float2 cur_point;
    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    cur_point.x = points1[n * P1 * 2 + p1 * 2];
    cur_point.y = points1[n * P1 * 2 + p1 * 2 + 1];

    float grid_min_x = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_MIN_X];
    float grid_min_y = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_MIN_Y];
    float grid_delta = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_DELTA];
    int grid_res_x = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_RES_X];
    int grid_res_y = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_RES_Y];
    int grid_total = params[n * GRID_2D_PARAMS_SIZE + GRID_2D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point.x - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point.y - grid_min_y - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point.x - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point.y - grid_min_y + cur_r) * grid_delta);
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        int cell_idx = x * grid_res_y + y;
        int p2_start = pc2_grid_off[n * G + cell_idx];
        int p2_end;
        if (cell_idx + 1 == grid_total) {
          p2_end = lengths2[n];
        } else {
          p2_end = pc2_grid_off[n * G + cell_idx + 1];
        }
        for (int p2 = p2_start; p2 < p2_end; ++p2) {
          diff.x = points2[n * P2 * 2 + p2 * 2] - cur_point.x;
          diff.y = points2[n * P2 * 2 + p2 * 2 + 1] - cur_point.y;
          sqdist = diff.x * diff.x + diff.y * diff.y;
          if (sqdist <= cur_r2) {
            mink.add(sqdist, sorted_points2_idxs[n * P2 + p2]);
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + old_p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + old_p1 * K + k] = min_dists[k];
    }
  }
}

/*
template <int K>
__global__ void FindNbrs3DKernel(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params, float *__restrict__ dists,
    long *__restrict__ idxs, int N, int P1, int P2, int G,
    const float *__restrict__ rs, const float *__restrict__ r2s) {
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
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float3 cur_point;
    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    cur_point.x = points1[n * P1 * 3 + p1 * 3];
    cur_point.y = points1[n * P1 * 3 + p1 * 3 + 1];
    cur_point.z = points1[n * P1 * 3 + p1 * 3 + 2];

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
    int grid_total = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point.x - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point.y - grid_min_y - cur_r) * grid_delta);
    int min_gc_z =
        (int)std::floor((cur_point.z - grid_min_z - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point.x - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point.y - grid_min_y + cur_r) * grid_delta);
    int max_gc_z =
        (int)std::floor((cur_point.z - grid_min_z + cur_r) * grid_delta);
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        for (int z = max(min_gc_z, 0); z <= min(max_gc_z, grid_res_z - 1);
             ++z) {
          int cell_idx = (x * grid_res_y + y) * grid_res_z + z;
          int p2_start = pc2_grid_off[n * G + cell_idx];
          int p2_end;
          if (cell_idx + 1 == grid_total) {
            p2_end = lengths2[n];
          } else {
            p2_end = pc2_grid_off[n * G + cell_idx + 1];
          }
          for (int p2 = p2_start; p2 < p2_end; ++p2) {
            diff.x = points2[n * P2 * 3 + p2 * 3] - cur_point.x;
            diff.y = points2[n * P2 * 3 + p2 * 3 + 1] - cur_point.y;
            diff.z = points2[n * P2 * 3 + p2 * 3 + 2] - cur_point.z;
            sqdist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            if (sqdist <= cur_r2) {
              mink.add(sqdist, sorted_points2_idxs[n * P2 + p2]);
            }
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + old_p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + old_p1 * K + k] = min_dists[k];
    }
  }
}
*/

__global__ void FindNbrsNDKernelV0(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params, float *__restrict__ dists,
    long *__restrict__ idxs, int N, int P1, int P2, int G, int D, int K,
    const float *__restrict__ rs, const float *__restrict__ r2s) {
  // access all the data in global memory directly
  float3 cur_point_3;
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    cur_point_3.x = points1[n * P1 * D + p1 * D + 0];
    cur_point_3.y = points1[n * P1 * D + p1 * D + 1];
    cur_point_3.z = points1[n * P1 * D + p1 * D + 2];

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
    int grid_total = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point_3.x - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point_3.y - grid_min_y - cur_r) * grid_delta);
    int min_gc_z =
        (int)std::floor((cur_point_3.z - grid_min_z - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point_3.x - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point_3.y - grid_min_y + cur_r) * grid_delta);
    int max_gc_z =
        (int)std::floor((cur_point_3.z - grid_min_z + cur_r) * grid_delta);
    int offset = n * P1 * K + old_p1 * K;
    MinK<float, long> mink(dists + offset, idxs + offset, K);
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        for (int z = max(min_gc_z, 0); z <= min(max_gc_z, grid_res_z - 1);
             ++z) {
          int cell_idx = (x * grid_res_y + y) * grid_res_z + z;
          int p2_start = pc2_grid_off[n * G + cell_idx];
          int p2_end;
          if (cell_idx + 1 == grid_total) {
            p2_end = lengths2[n];
          } else {
            p2_end = pc2_grid_off[n * G + cell_idx + 1];
          }
          for (int p2 = p2_start; p2 < p2_end; ++p2) {
            float sqdist = 0;
            float diff;
            for (int d = 0; d < D; ++d) {
              diff = points2[n * P2 * D + p2 * D + d] -
                     points1[n * P1 * D + p1 * D + d];
              sqdist += diff * diff;
            }
            if (sqdist <= cur_r2) {
              mink.add(sqdist, sorted_points2_idxs[n * P2 + p2]);
            }
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    // for (int k = 0; k < mink.size(); ++k) {
    //   idxs[n * P1 * K + old_p1 * K + k] = min_idxs[k];
    //   dists[n * P1 * K + old_p1 * K + k] = min_dists[k];
    // }
  }
}

template <int D>
__global__ void FindNbrsNDKernelV1(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params, float *__restrict__ dists,
    long *__restrict__ idxs, int N, int P1, int P2, int G, int K,
    const float *__restrict__ rs, const float *__restrict__ r2s) {
  float cur_point[D];

  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
    int grid_total = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point[0] - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point[1] - grid_min_y - cur_r) * grid_delta);
    int min_gc_z =
        (int)std::floor((cur_point[2] - grid_min_z - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point[0] - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point[1] - grid_min_y + cur_r) * grid_delta);
    int max_gc_z =
        (int)std::floor((cur_point[2] - grid_min_z + cur_r) * grid_delta);
    int offset = n * P1 * K + old_p1 * K;
    MinK<float, long> mink(dists + offset, idxs + offset, K);
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        for (int z = max(min_gc_z, 0); z <= min(max_gc_z, grid_res_z - 1);
             ++z) {
          int cell_idx = (x * grid_res_y + y) * grid_res_z + z;
          int p2_start = pc2_grid_off[n * G + cell_idx];
          int p2_end;
          if (cell_idx + 1 == grid_total) {
            p2_end = lengths2[n];
          } else {
            p2_end = pc2_grid_off[n * G + cell_idx + 1];
          }
          for (int p2 = p2_start; p2 < p2_end; ++p2) {
            float sqdist = 0;
            float diff;
            for (int d = 0; d < D; ++d) {
              diff = points2[n * P2 * D + p2 * D + d] - cur_point[d];
              sqdist += diff * diff;
            }
            if (sqdist <= cur_r2) {
              mink.add(sqdist, sorted_points2_idxs[n * P2 + p2]);
            }
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    // for (int k = 0; k < mink.size(); ++k) {
    //   idxs[n * P1 * K + old_p1 * K + k] = min_idxs[k];
    //   dists[n * P1 * K + old_p1 * K + k] = min_dists[k];
    // }
  }
}

template <int D, int K>
__global__ void FindNbrsNDKernelV2(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const long *__restrict__ lengths1, const long *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params, float *__restrict__ dists,
    long *__restrict__ idxs, int N, int P1, int P2, int G,
    const float *__restrict__ rs, const float *__restrict__ r2s) {
  float min_dists[K];
  int min_idxs[K];
  float cur_point[D];

  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    float cur_r = rs[n];
    float cur_r2 = r2s[n];
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
    int grid_total = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point[0] - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point[1] - grid_min_y - cur_r) * grid_delta);
    int min_gc_z =
        (int)std::floor((cur_point[2] - grid_min_z - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point[0] - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point[1] - grid_min_y + cur_r) * grid_delta);
    int max_gc_z =
        (int)std::floor((cur_point[2] - grid_min_z + cur_r) * grid_delta);
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        for (int z = max(min_gc_z, 0); z <= min(max_gc_z, grid_res_z - 1);
             ++z) {
          int cell_idx = (x * grid_res_y + y) * grid_res_z + z;
          int p2_start = pc2_grid_off[n * G + cell_idx];
          int p2_end;
          if (cell_idx + 1 == grid_total) {
            p2_end = lengths2[n];
          } else {
            p2_end = pc2_grid_off[n * G + cell_idx + 1];
          }
          for (int p2 = p2_start; p2 < p2_end; ++p2) {
            float sqdist = 0;
            float diff;
            for (int d = 0; d < D; ++d) {
              diff = points2[n * P2 * D + p2 * D + d] - cur_point[d];
              sqdist += diff * diff;
            }
            if (sqdist <= cur_r2) {
              mink.add(sqdist, sorted_points2_idxs[n * P2 + p2]);
            }
          }
        }
      }
    }
    // TODO: add return_sort here
    mink.sort();
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + old_p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + old_p1 * K + k] = min_dists[k];
    }
  }
}

template <int D>
struct FindNbrsKernelV1Functor {
  static void run(int blocks, int threads,
                  const float *__restrict__ points1,            // (N, P1, D)
                  const float *__restrict__ points2,            // (N, P2, D)
                  const long *__restrict__ lengths1,            // (N,)
                  const long *__restrict__ lengths2,            // (N,)
                  const int *__restrict__ pc2_grid_off,         // (N, G)
                  const int *__restrict__ sorted_points1_idxs,  // (N, P)
                  const int *__restrict__ sorted_points2_idxs,  // (N, P)
                  const float *__restrict__ params,             // (N,)
                  float *__restrict__ dists,                    // (N, P1, K)
                  long *__restrict__ idxs,                      // (N, P1, K)
                  int N, int P1, int P2, int G, int K, const float *rs,
                  const float *r2s) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (D == 2) {
      FindNbrs2DKernelV1<<<blocks, threads, 0, stream>>>(
          points1, points2, lengths1, lengths2, pc2_grid_off,
          sorted_points1_idxs, sorted_points2_idxs, params, dists, idxs, N, P1,
          P2, G, K, rs, r2s);
    } else {
      // FindNbrs3DKernel<K><<<blocks, threads, 0, stream>>>(
      //     points1, points2, lengths1, lengths2, pc2_grid_off,
      //     sorted_points1_idxs, sorted_points2_idxs, params, dists, idxs, N,
      //     P1, P2, G, rs, r2s);
      FindNbrsNDKernelV1<D><<<blocks, threads, 0, stream>>>(
          points1, points2, lengths1, lengths2, pc2_grid_off,
          sorted_points1_idxs, sorted_points2_idxs, params, dists, idxs, N, P1,
          P2, G, K, rs, r2s);
    }
  }
};

template <int D, int K>
struct FindNbrsKernelV2Functor {
  static void run(int blocks, int threads,
                  const float *__restrict__ points1,            // (N, P1, D)
                  const float *__restrict__ points2,            // (N, P2, D)
                  const long *__restrict__ lengths1,            // (N,)
                  const long *__restrict__ lengths2,            // (N,)
                  const int *__restrict__ pc2_grid_off,         // (N, G)
                  const int *__restrict__ sorted_points1_idxs,  // (N, P)
                  const int *__restrict__ sorted_points2_idxs,  // (N, P)
                  const float *__restrict__ params,             // (N,)
                  float *__restrict__ dists,                    // (N, P1, K)
                  long *__restrict__ idxs,                      // (N, P1, K)
                  int N, int P1, int P2, int G, const float *rs,
                  const float *r2s) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (D == 2) {
      FindNbrs2DKernelV2<K><<<blocks, threads, 0, stream>>>(
          points1, points2, lengths1, lengths2, pc2_grid_off,
          sorted_points1_idxs, sorted_points2_idxs, params, dists, idxs, N, P1,
          P2, G, rs, r2s);
    } else {
      // FindNbrs3DKernel<K><<<blocks, threads, 0, stream>>>(
      //     points1, points2, lengths1, lengths2, pc2_grid_off,
      //     sorted_points1_idxs, sorted_points2_idxs, params, dists, idxs, N,
      //     P1, P2, G, rs, r2s);
      FindNbrsNDKernelV2<D, K><<<blocks, threads, 0, stream>>>(
          points1, points2, lengths1, lengths2, pc2_grid_off,
          sorted_points1_idxs, sorted_points2_idxs, params, dists, idxs, N, P1,
          P2, G, rs, r2s);
    }
  }
};

bool InBounds(const int min, const int x, const int max) {
  return min <= x && x <= max;
}

int FRNNChooseVersion(const int D, const int K) {
  if (InBounds(V2_MIN_D, D, V2_MAX_D) && InBounds(V2_MIN_K, K, V2_MAX_K)) {
    return 2;
  } else if (InBounds(V1_MIN_D, D, V1_MAX_D)) {
    return 1;
  } else if (InBounds(V0_MIN_D, D, V0_MAX_D)) {
    return 0;
  } else {
    return -1;
  }
}

std::tuple<at::Tensor, at::Tensor> FindNbrsCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2,
    const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs, const at::Tensor params, int K,
    const at::Tensor rs, const at::Tensor r2s) {
  at::TensorArg points1_t{points1, "points1", 1};
  at::TensorArg points2_t{points2, "points2", 2};
  at::TensorArg lengths1_t{lengths1, "lengths1", 3};
  at::TensorArg lengths2_t{lengths2, "lengths2", 4};
  at::TensorArg pc2_grid_off_t{pc2_grid_off, "pc2_grid_off", 5};
  at::TensorArg sorted_points1_idxs_t{sorted_points1_idxs,
                                      "sorted_points1_idxs", 6};
  at::TensorArg sorted_points2_idxs_t{sorted_points2_idxs,
                                      "sorted_points2_idxs", 7};
  at::TensorArg params_t{params, "params", 8};
  at::TensorArg rs_t{rs, "rs", 10};
  at::TensorArg r2s_t{r2s, "r2s", 11};

  at::CheckedFrom c = "FindNbrsCUDA";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, pc2_grid_off_t,
          sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t, r2s_t});
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t, r2s_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  at::checkAllSameType(
      c, {pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int N = points1.size(0);
  int P1 = points1.size(1);
  int D = points1.size(2);
  int P2 = points2.size(1);
  int G = pc2_grid_off.size(1);

  auto idxs = at::full({N, P1, K}, -1, lengths1.options());
  auto dists = at::full({N, P1, K}, -1, points1.options());

  int blocks = 256;
  int threads = 256;

  int version = FRNNChooseVersion(D, K);

  // DispatchKernel1D<FindNbrsKernelFunctor, MIN_K, MAX_K>(
  //     K, blocks, threads, D, points1.contiguous().data_ptr<float>(),
  //     points2.contiguous().data_ptr<float>(),
  //     lengths1.contiguous().data_ptr<long>(),
  //     lengths2.contiguous().data_ptr<long>(),
  //     pc2_grid_off.contiguous().data_ptr<int>(),
  //     sorted_points1_idxs.contiguous().data_ptr<int>(),
  //     sorted_points2_idxs.contiguous().data_ptr<int>(),
  //     params.contiguous().data_ptr<float>(), dists.data_ptr<float>(),
  //     idxs.data_ptr<long>(), N, P1, P2, G, rs.data_ptr<float>(),
  //     r2s.data_ptr<float>());
  if (version == 0) {
    assert(D > 2);
    FindNbrsNDKernelV0<<<blocks, threads, 0, stream>>>(
        points1.contiguous().data_ptr<float>(),
        points2.contiguous().data_ptr<float>(),
        lengths1.contiguous().data_ptr<long>(),
        lengths2.contiguous().data_ptr<long>(),
        pc2_grid_off.contiguous().data_ptr<int>(),
        sorted_points1_idxs.contiguous().data_ptr<int>(),
        sorted_points2_idxs.contiguous().data_ptr<int>(),
        params.contiguous().data_ptr<float>(), dists.data_ptr<float>(),
        idxs.data_ptr<long>(), N, P1, P2, G, D, K, rs.data_ptr<float>(),
        r2s.data_ptr<float>());
  } else if (version == 1) {
    DispatchKernel1D<FindNbrsKernelV1Functor, V1_MIN_D, V1_MAX_D>(
        D, blocks, threads, points1.contiguous().data_ptr<float>(),
        points2.contiguous().data_ptr<float>(),
        lengths1.contiguous().data_ptr<long>(),
        lengths2.contiguous().data_ptr<long>(),
        pc2_grid_off.contiguous().data_ptr<int>(),
        sorted_points1_idxs.contiguous().data_ptr<int>(),
        sorted_points2_idxs.contiguous().data_ptr<int>(),
        params.contiguous().data_ptr<float>(), dists.data_ptr<float>(),
        idxs.data_ptr<long>(), N, P1, P2, G, K, rs.data_ptr<float>(),
        r2s.data_ptr<float>());
  } else if (version == 2) {
    DispatchKernel2D<FindNbrsKernelV2Functor, V2_MIN_D, V2_MAX_D, V2_MIN_K,
                     V2_MAX_K>(
        D, K, blocks, threads, points1.contiguous().data_ptr<float>(),
        points2.contiguous().data_ptr<float>(),
        lengths1.contiguous().data_ptr<long>(),
        lengths2.contiguous().data_ptr<long>(),
        pc2_grid_off.contiguous().data_ptr<int>(),
        sorted_points1_idxs.contiguous().data_ptr<int>(),
        sorted_points2_idxs.contiguous().data_ptr<int>(),
        params.contiguous().data_ptr<float>(), dists.data_ptr<float>(),
        idxs.data_ptr<long>(), N, P1, P2, G, rs.data_ptr<float>(),
        r2s.data_ptr<float>());
  } else {
    AT_ASSERTM(false, "Invalid version for find_nbrs");
  }

  return std::make_tuple(idxs, dists);
}