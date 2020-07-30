#include <torch/extension.h>
std::tuple<at::Tensor, at::Tensor> TestGrid(
    const at::Tensor& Points, int K, float r);