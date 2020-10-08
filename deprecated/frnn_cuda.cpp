#include <tuple>
#include "utils/check.h"
#include "grid.h"

std::tuple<at::Tensor, at::Tensor> FRNNBruteForceCUDA(
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

    return FRNNBruteForceCUDA(p1, p2, lengths1, lengths2, K, rsq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frnn_bf_gpu", &FRNNBruteForceGpu, "Brute Force Fixed Radius Nearest Neighbor Search on GPU");
    m.def("grid_test_gpu", &TestGridCUDA, "Grid Test on GPU");
    m.def("prefix_sum", &PrefixSum);
}