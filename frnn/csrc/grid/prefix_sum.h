#pragma once
#include <ATen/ATen.h>
#include "grid.h"

/*
at::Tensor PrefixSumCUDA(
    const at::Tensor grid_cnt,
    const GridParams* params);
at::Tensor TestPrefixSumCUDA(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r);
*/

at::Tensor PrefixSumCUDA(
    const at::Tensor grid_cnt,
    const at::Tensor params);

at::Tensor PrefixSumCPU(
    const at::Tensor grid_cnt,
    const GridParams* params);


at::Tensor TestPrefixSumCPU(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r);