#include <torch/extension.h>
#include "grid.cuh"

at::Tensor TestGrid(
    const at::Tensor Points, int K, float r);
std::tuple<at::Tensor, at::Tensor> TestGridCUDA(
        const at::Tensor Points,
        const at::Tensor bbox_max,
        const at::Tensor bbox_min,
        int K,
        float r);
 
void SetupGridParams(
    const at::Tensor Points,
    float cell_size,
    GridParams& params);
 
 
