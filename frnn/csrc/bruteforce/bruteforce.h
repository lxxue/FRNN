#include <ATen/ATen.h>
#include <tuple>

/* fixed radius nearest neighbor search on GPU using exhaustive O(n^2) method */
std::tuple<at::Tensor, at::Tensor>
FRNNBruteForceCUDA(const at::Tensor &p1, const at::Tensor &p2,
                   const at::Tensor &lengths1, const at::Tensor &lengths2,
                   int K, float r);

/* fixed radius nearest neighbor search on CPU using brute force O(n^2) method
 * used as baseline & ground truth. based on pytorch3d KNearestNeighborIdxCpu */
std::tuple<at::Tensor, at::Tensor> FRNNBruteForceCPU(const at::Tensor &p1,
                                                     const at::Tensor &p2,
                                                     const at::Tensor &lengths1,
                                                     const at::Tensor &lengths2,
                                                     int K, float r);