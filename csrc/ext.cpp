#include <torch/extension.h>
#include "grid/grid.h"
#include "grid/prefix_sum.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup_grid_params", &SetupGridParams);
  m.def("test_setup_grid_params_cuda", &TestSetupGridParamsCUDA);
  m.def("insert_points_int_cuda", &InsertPointsCUDA<int>);
  m.def("test_insert_points_cuda", &TestInsertPointsCUDA);
  m.def("test_insert_points_cpu", &TestInsertPointsCPU);
  m.def("prefix_sum_cuda", &PrefixSumCUDA);
  m.def("prefix_sum_cpu", &PrefixSumCPU);
}