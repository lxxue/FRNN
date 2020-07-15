#include <torch/extension.h>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x "must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> FRNNBruteForceCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float rsq);
    
 std::tuple<at::Tensor, at::Tensor> FRNNBruteForceGpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    int K,
    float rsq) {
    
    CHECK_CONTIGUOUS_CUDA(p1);
    CHECK_CONTIGUOUS_CUDA(p2);
    CHECK_CONTIGUOUS_CUDA(lengths1);
    CHECK_CONTIGUOUS_CUDA(lengths2);

    return FRNNBruteForceCuda(p1, p2, lengths1, lengths2, K, rsq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frnn_bf_gpu", &FRNNBruteForceGpu, "Brute Force Fixed Radius Nearest Neighbor Search on GPU");
}