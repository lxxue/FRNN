#include <ATen/ATen.h>

at::Tensor PrefixSumCUDA(
    const at::Tensor grid_cnt);

at::Tensor PrefixSumCPU(
    const at::Tensor grid_cnt);