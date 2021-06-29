#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define GRID_3D_MIN_X 0
#define GRID_3D_MIN_Y 1
#define GRID_3D_MIN_Z 2
#define GRID_3D_DELTA 3
#define GRID_3D_RES_X 4
#define GRID_3D_RES_Y 5
#define GRID_3D_RES_Z 6
#define GRID_3D_TOTAL 7
#define GRID_3D_PARAMS_SIZE 8
#define GRID_3D_MAX_RES 64

#define GRID_2D_MIN_X 0
#define GRID_2D_MIN_Y 1
#define GRID_2D_DELTA 2
#define GRID_2D_RES_X 3
#define GRID_2D_RES_Y 4
#define GRID_2D_TOTAL 5
#define GRID_2D_PARAMS_SIZE 6
#define GRID_2D_MAX_RES 512

// TODO: Optimize for large K
constexpr int V0_MIN_D = 2;
constexpr int V0_MAX_D = 1024;

constexpr int V1_MIN_D = 2;
constexpr int V1_MAX_D = 32;

constexpr int V2_MIN_D = 2;
constexpr int V2_MAX_D = 8;
constexpr int V2_MIN_K = 1;
constexpr int V2_MAX_K = 32;

// now use at::Tensor to store grid params
// and we setup grid params in python
// this struct and corresponding CPU function are now for validation only
struct GridParams {
  float3 grid_size, grid_min, grid_max;
  int3 grid_res;
  int grid_total;
  // 1/grid_cell_size, used to convert position to cell index via multiplication
  float grid_delta;

  GridParams() {}
};

void SetupGridParams(float *bboxes, float cell_size, GridParams *params);
