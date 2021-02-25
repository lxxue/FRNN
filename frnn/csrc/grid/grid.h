#pragma once
#include "utils/cutil_math.h"

#define GRID_3D_MIN_X 0
#define GRID_3D_MIN_Y 1
#define GRID_3D_MIN_Z 2
#define GRID_3D_DELTA 3
#define GRID_3D_RES_X 4
#define GRID_3D_RES_Y 5
#define GRID_3D_RES_Z 6
#define GRID_3D_TOTAL 7
#define GRID_3D_PARAMS_SIZE 8
#define GRID_3D_MAX_RES 128

#define GRID_2D_MIN_X 0
#define GRID_2D_MIN_Y 1
#define GRID_2D_DELTA 2
#define GRID_2D_RES_X 3
#define GRID_2D_RES_Y 4
#define GRID_2D_TOTAL 5
#define GRID_2D_PARAMS_SIZE 6
#define GRID_2D_MAX_RES 1024

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

void SetupGridParams(
    float* bboxes,
    float cell_size,
    GridParams* params);

// TODO: add docs
void InsertPointsCUDA(
    const at::Tensor points,    // (N, P, 2/3)
    const at::Tensor lengths,   // (N,)
    const at::Tensor params,    // (N, 6/8)
    at::Tensor grid_cnt,        // (N, G)
    at::Tensor grid_cell,       // (N, P)      
    at::Tensor grid_idx,        // (N, P)
    int G);

void InsertPointsCPU(
    const at::Tensor points, 
    const at::Tensor lengths, 
    at::Tensor grid, 
    at::Tensor grid_cnt, 
    at::Tensor grid_cell, 
    at::Tensor grid_next, 
    at::Tensor grid_idx,
    GridParams* params);

std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCPU(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r);

// TODO: add docs
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
    const at::Tensor r2s);
 
std::tuple<at::Tensor, at::Tensor> FindNbrsCPU(
    const at::Tensor points1,          // (N, P1, 2/3)
    const at::Tensor points2,          // (N, P2, 2/3)
    const at::Tensor lengths1,         // (N,)
    const at::Tensor lengths2,         // (N,)
    const at::Tensor grid_off,         // (N, G)
    const at::Tensor sorted_point_idx, // (N, P2)
    const GridParams *params,
    int K,
    float r); 

std::tuple<at::Tensor, at::Tensor> TestFindNbrsCPU(
    const at::Tensor bboxes,
    const at::Tensor points1,
    const at::Tensor points2,
    const at::Tensor lengths1,
    const at::Tensor lengths2,
    int K,
    float r);