#pragma once
#include "grid.h"
#include <ATen/ATen.h>

at::Tensor PrefixSumCUDA(const at::Tensor grid_cnt, const at::Tensor params);

at::Tensor PrefixSumCPU(const at::Tensor grid_cnt, const GridParams *params);

at::Tensor TestPrefixSumCPU(const at::Tensor bboxes, const at::Tensor points,
                            const at::Tensor lengths, float r);