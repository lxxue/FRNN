#include <ATen/ATen.h>

// a simple CPU prefix
at::Tensor PrefixSumCPU(
    const at::Tensor lengths,
    const at::Tensor grid_cnt) {
  int N = lengths.size(0); 
  int max_grid_total = grid_cnt.size(1);
  
  auto lengths_a = lengths.accessor<int, 1>();
  auto grid_cnt_a = grid_cnt.accessor<int, 2>();

  at::Tensor grid_off = at::zeros({N, max_grid_total}, grid_cnt.options());
  auto grid_off_a = grid_off.accessor<int, 2>();
  for (int n = 0; n < N; ++n) {
    for (int i = 1; i < lengths_a[n]; ++i) {
      grid_off_a[n][i] = grid_off_a[n][i-1] + grid_cnt_a[n][i-1];
    }
  }
  return grid_off;
}