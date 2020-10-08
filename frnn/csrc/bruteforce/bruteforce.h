#include <ATen/ATen.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> FRNNBruteForceCUDA(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float r);

std::tuple<at::Tensor, at::Tensor> FRNNBruteForceCPU(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1, 
    const at::Tensor& lengths2,
    int K,
    float r);