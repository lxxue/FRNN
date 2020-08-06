#include <ATen/ATen.h>

at::Tensor PrefixSumCUDA(
    const at::Tensor lengths,
    const at::Tensor grid_cnt);

at::Tensor PrefixSumCPU(
    const at::Tensor lengths,
    const at::Tensor grid_cnt);