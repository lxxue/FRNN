#pragma once
#include <torch/extension.h>

// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor.")
// #define CHECK_CONTIGUOUS(x) \
//   TORCH_CHECK(x.is_contiguous(), #x "must be contiguous.")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), "must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)
