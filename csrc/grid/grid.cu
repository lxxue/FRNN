#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "grid.h"

void SetupGridParamsCUDA (
    float* points_min,
    float* points_max,
    float cell_size,
    GridParams* params) {
  params->grid_min.x = points_min[0];
  params->grid_min.y = points_min[1];
  params->grid_min.z = points_min[2];
  params->grid_max.x = points_max[0];
  params->grid_max.y = points_max[1];
  params->grid_max.z = points_max[2];

  return;
}
