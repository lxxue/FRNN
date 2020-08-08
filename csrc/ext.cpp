#include <torch/extension.h>
#include "grid/grid.h"
#include "grid/prefix_sum.h"
#include "grid/counting_sort.h"
#include "bruteforce/bruteforce.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup_grid_params", &SetupGridParams);
  m.def("test_setup_grid_params_cuda", &TestSetupGridParamsCUDA);

  m.def("insert_points_int_cuda", &InsertPointsCUDA);
  m.def("test_insert_points_cuda", &TestInsertPointsCUDA);
  m.def("test_insert_points_cpu", &TestInsertPointsCPU);
  
  m.def("prefix_sum_cuda", &PrefixSumCUDA);
  m.def("prefix_sum_cpu", &PrefixSumCPU);
  m.def("test_prefix_sum_cuda", &TestPrefixSumCUDA);
  m.def("test_prefix_sum_cpu", &TestPrefixSumCPU);

  m.def("counting_sort_cuda", &CountingSortCUDA);
  m.def("counting_sort_cpu", &CountingSortCPU);
  
  m.def("find_nbrs_cuda", &FindNbrsCUDA);
  m.def("test_find_nbrs_cuda", &TestFindNbrsCUDA);
  
  m.def("frnn_bf_cuda", &FRNNBruteForceCUDA);
}