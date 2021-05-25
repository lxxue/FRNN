#include <torch/extension.h>

#include "backward/backward.h"
#include "bruteforce/bruteforce.h"
#include "grid/counting_sort.h"
#include "grid/find_nbrs.h"
#include "grid/grid.h"
#include "grid/insert_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup_grid_params", &SetupGridParams);

  m.def("insert_points_cuda", &InsertPointsCUDA);
  m.def("test_insert_points_cpu", &TestInsertPointsCPU);

  m.def("counting_sort_cuda", &CountingSortCUDA);
  m.def("counting_sort_cpu", &CountingSortCPU);

  m.def("find_nbrs_cuda", &FindNbrsCUDA);
  m.def("find_nbrs_cpu", &FindNbrsCPU);
  m.def("test_find_nbrs_cpu", &TestFindNbrsCPU);

  m.def("frnn_bf_cuda", &FRNNBruteForceCUDA);
  m.def("frnn_bf_cpu", &FRNNBruteForceCPU);

  m.def("frnn_backward_cuda", &FRNNBackwardCUDA);
}
