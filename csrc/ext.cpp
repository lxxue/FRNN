#include <torch/extension.h>
#include "grid.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup_grid_params_cuda", &SetupGridPramsCUDA);
}