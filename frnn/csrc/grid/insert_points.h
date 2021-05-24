#include "grid/grid.h"
#include "utils/dispatch.h"

// TODO: add docs
void InsertPointsCUDA(const at::Tensor points,   // (N, P, 2/3)
                      const at::Tensor lengths,  // (N,)
                      const at::Tensor params,   // (N, 6/8)
                      at::Tensor grid_cnt,       // (N, G)
                      at::Tensor grid_cell,      // (N, P)
                      at::Tensor grid_idx,       // (N, P)
                      int G);

void InsertPointsCPU(const at::Tensor points, const at::Tensor lengths,
                     at::Tensor grid, at::Tensor grid_cnt, at::Tensor grid_cell,
                     at::Tensor grid_next, at::Tensor grid_idx,
                     GridParams *params);

std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCPU(
    const at::Tensor bboxes, const at::Tensor points, const at::Tensor lengths,
    float r);
