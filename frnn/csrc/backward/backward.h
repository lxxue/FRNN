#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "grid/grid.h"
#include "utils/dispatch.h"

std::tuple<at::Tensor, at::Tensor> FRNNBackwardCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lenghts1, const at::Tensor lengths2, const at::Tensor idxs,
    const at::Tensor grad_dists);