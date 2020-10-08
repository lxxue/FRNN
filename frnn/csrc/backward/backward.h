#pragma once
#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor> FRNNBackwardCUDA(
    const at::Tensor points1,
    const at::Tensor points2,
    const at::Tensor lenghts1,
    const at::Tensor lengths2,
    const at::Tensor idxs,
    const at::Tensor grad_dists);