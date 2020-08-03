#include "utils/cutil_math.h"

struct GridParams {
  float3 grid_size, grid_min, grid_max;
  int3 grid_res;
  int grid_total;
  // 1/grid_cell_size, used to convert position to cell index via multiplication
  float grid_delta; 

  GridParams() {}
};

void SetupGridParamsCUDA (
    float* bboxes,
    float cell_size,
    GridParams* params);


void TestSetupGridParamsCUDA (
    at::Tensor bboxes,
    float r);