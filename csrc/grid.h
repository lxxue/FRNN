#pragma once
#include <torch/extension.h>
#include "utils/math.h"

struct GridParams {
    float3  gridSize, gridMin, gridMax;
    int3    gridRes, gridScanMax;
    int     gridSrch, gridTotal; // , gridAdjCnt, gridActive;
    // int     gridAdj[125];
    float   gridCellSize, gridDelta;

    GridParams() {}
};


std::tuple<at::Tensor, at::Tensor> TestGrid(
    const at::Tensor& Points, int K, float r);