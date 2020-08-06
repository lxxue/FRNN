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

void InsertPointsCUDA(
  const at::Tensor points,  
  const at::Tensor lengths, 
  at::Tensor grid_cnt,      
  at::Tensor grid_cell,     
  at::Tensor grid_idx,      
  const GridParams* params);

std::tuple<at::Tensor, at::Tensor> TestInsertPointsCUDA(
   const at::Tensor bboxes,  
   const at::Tensor points,  
   const at::Tensor lengths,
   float r);


std::tuple<at::Tensor, at::Tensor> TestInsertPointsCPU(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r);