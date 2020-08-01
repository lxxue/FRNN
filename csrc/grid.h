#include <torch/extension.h>
#include "utils/math.h"
#include "grid.cuh"

at::Tensor TestGrid(
    const at::Tensor Points, int K, float r);
at::Tensor TestGridCUDA(
        const at::Tensor Points,
        const at::Tensor bbox_max,
        const at::Tensor bbox_min,
        int K,
        float r);
at::Tensor TensorTest(at::Tensor Points);
