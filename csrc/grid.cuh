
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include "utils/cutil_math.h"

#ifndef GRID_H
#define GRID_H
struct GridParams {
    float3  gridSize, gridMin, gridMax;
    int3    gridRes, gridScanMax;
    int     gridSrch, gridTotal; // , gridAdjCnt, gridActive;
    // int     gridAdj[125];
    float   gridCellSize, gridDelta;

    GridParams() {}
};


void SetupGridParams(
    const at::Tensor Points,
    float cell_size,
    GridParams& params);

void SetupGridParamsCUDA(
    float* bbox_max,
    float* bbox_min,
    float cell_size,
    GridParams& params);

#endif
