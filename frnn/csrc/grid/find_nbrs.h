#include <tuple>

#include "grid/counting_sort.h"
#include "grid/grid.h"
#include "utils/mink.cuh"
// customized dispatch utils for our function type
#include "utils/dispatch.h"

// TODO: add docs
std::tuple<at::Tensor, at::Tensor> FindNbrsCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2,
    const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs, const at::Tensor params, int K,
    const at::Tensor rs, const at::Tensor r2s);

std::tuple<at::Tensor, at::Tensor> FindNbrsCPU(
    const at::Tensor points1,           // (N, P1, 2/3)
    const at::Tensor points2,           // (N, P2, 2/3)
    const at::Tensor lengths1,          // (N,)
    const at::Tensor lengths2,          // (N,)
    const at::Tensor grid_off,          // (N, G)
    const at::Tensor sorted_point_idx,  // (N, P2)
    const GridParams *params, int K, float r);

std::tuple<at::Tensor, at::Tensor> TestFindNbrsCPU(
    const at::Tensor bboxes, const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2, int K, float r);