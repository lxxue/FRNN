#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "grid/grid.h"
#include "utils/dispatch.h"

void CountingSortCUDA(const at::Tensor points, const at::Tensor lengths,
                      const at::Tensor grid_cell, const at::Tensor grid_idx,
                      const at::Tensor grid_off, at::Tensor sorted_points,
                      at::Tensor sorted_point_idx);

void CountingSortCPU(const at::Tensor points, const at::Tensor lengths,
                     const at::Tensor grid_cell, const at::Tensor grid_idx,
                     const at::Tensor grid_off, at::Tensor sorted_points,
                     at::Tensor sorted_point_idx);