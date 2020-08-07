#pragma once
#include "utils/cutil_math.h"

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


void TestSetupGridParamsCUDA(
    at::Tensor bboxes,
    float r);

template<typename idx_t>
void InsertPointsCUDA(
    const at::Tensor points,  
    const at::Tensor lengths, 
    at::Tensor grid_cnt,      
    at::Tensor grid_cell,     
    at::Tensor grid_idx,      
    int G,
    const GridParams* params);


std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCUDA(
   const at::Tensor bboxes,  
   const at::Tensor points,  
   const at::Tensor lengths,
   float r);


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

std::tuple<at::Tensor, at::Tensor> FindNbrsCUDA(
    const at::Tensor points1,
    const at::Tensor points2,
    const at::Tensor lengths1,
    const at::Tensor lengths2,
    const at::Tensor grid_off,
    const at::Tensor sorted_point_idx,
    const GridParams* params,
    int K,
    float r);

  
std::tuple<at::Tensor, at::Tensor> TestFindNbrsCUDA(
    const at::Tensor bboxes,  
    const at::Tensor points1,  
    const at::Tensor points2,
    const at::Tensor lengths1,
    const at::Tensor lengths2,
    int K,
    float r);