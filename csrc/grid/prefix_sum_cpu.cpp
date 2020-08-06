#include <ATen/ATen.h>

// a simple CPU prefix
at::Tensor PrefixSumCPU(
    const at::Tensor grid_cnt) {
  int N = grid_cnt.size(0); 
  int G = grid_cnt.size(1);
  
  auto grid_cnt_a = grid_cnt.accessor<int, 2>();

  at::Tensor grid_off = at::zeros({N, G}, grid_cnt.options());
  auto grid_off_a = grid_off.accessor<int, 2>();
  for (int n = 0; n < N; ++n) {
    for (int p = 1; p < G; ++p) {
      grid_off_a[n][p] = grid_off_a[n][p-1] + grid_cnt_a[n][p-1];
    }
  }
  return grid_off;
}