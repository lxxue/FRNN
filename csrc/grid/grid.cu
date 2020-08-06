#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>

#include "grid.h"

void SetupGridParams(
    float* bboxes,
    float cell_size,
    GridParams* params) {
  params->grid_min.x = bboxes[0];
  params->grid_max.x = bboxes[1];
  params->grid_min.y = bboxes[2];
  params->grid_max.y = bboxes[3];
  params->grid_min.z = bboxes[4];
  params->grid_max.z = bboxes[5];

  params->grid_size = params->grid_max - params->grid_min;
  params->grid_res.x = (int)(params->grid_size.x / cell_size) + 1;
  params->grid_res.y = (int)(params->grid_size.y / cell_size) + 1;
  params->grid_res.z = (int)(params->grid_size.z / cell_size) + 1;
  params->grid_total = params->grid_res.x * params->grid_res.y * params->grid_res.z;

  params->grid_delta = 1 / cell_size;

  return;
}

void TestSetupGridParamsCUDA(
    at::Tensor bboxes,  // N x 3 x 2 (min, max) at last dimension
    float r) {
  int N = bboxes.size(0);
  // TODO: cell_size determined joint by search radius and bbox_size
  // TODO: cell_size different for different point clouds in the batch
  float cell_size = r;
  // std::cout << "cudaMalloc done" << std::endl;
  GridParams* h_params = new GridParams[N];
  for (int i = 0; i < N; ++i) {
    SetupGridParams(
      bboxes.contiguous().data_ptr<float>() + i*6,
      cell_size,
      &h_params[i]
    );
    // std::cout << h_params[i].grid_min.x << ' ' << h_params[i].grid_min.y << ' ' << h_params[i].grid_min.z << std::endl;
    // std::cout << h_params[i].grid_max.x << ' ' << h_params[i].grid_max.y << ' ' << h_params[i].grid_max.z << std::endl;
    // std::cout << h_params[i].grid_size.x << ' ' << h_params[i].grid_size.y << ' ' << h_params[i].grid_size.z << std::endl;
    // std::cout << h_params[i].grid_res.x << ' ' << h_params[i].grid_res.y << ' ' << h_params[i].grid_res.z << std::endl;
    // std::cout << h_params[i].grid_total << ' ' << h_params[i].grid_delta << ' ' << std::endl; 
  }

  // std::cout << "Setup done" << std::endl;

  GridParams* d_params;
  cudaMalloc((void**)&d_params, N*sizeof(GridParams));
  cudaMemcpy(d_params, h_params, N*sizeof(GridParams), cudaMemcpyHostToDevice);

  GridParams* h_d_params = new GridParams[N];
  cudaMemcpy(h_d_params, d_params, N*sizeof(GridParams), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) {
    std::cout << h_d_params[i].grid_min.x << ' ' << h_d_params[i].grid_min.y << ' ' << h_d_params[i].grid_min.z << std::endl;
    std::cout << h_d_params[i].grid_max.x << ' ' << h_d_params[i].grid_max.y << ' ' << h_d_params[i].grid_max.z << std::endl;
    std::cout << h_d_params[i].grid_res.x << ' ' << h_d_params[i].grid_res.y << ' ' << h_d_params[i].grid_res.z << std::endl;
    std::cout << h_d_params[i].grid_total << ' ' << h_d_params[i].grid_delta << ' ' << std::endl; 
    std::cout << h_d_params[i].grid_size.x << ' ' << h_d_params[i].grid_size.y << ' ' << h_d_params[i].grid_size.z << std::endl;
  }
  delete[] h_params;
  delete[] h_d_params;
  cudaFree(d_params);
}

template <typename idx_t>
__global__ void InsertPointsKernel(
    const float* __restrict__ points,
    const idx_t* __restrict__ lengths,
    idx_t* grid_cnt, // not sure if we can use __restrict__ here
    idx_t* __restrict__ grid_cell,
    idx_t* __restrict__ grid_idx,
    size_t N,
    size_t P,
    size_t G,
    const GridParams* params) {


  const idx_t chunks_per_cloud = (1 + (P - 1) / blockDim.x);
  const idx_t chunks_to_do = N * chunks_per_cloud;
  for (idx_t chunk=blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const idx_t n = chunk / chunks_per_cloud;
    const idx_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    idx_t p = start_point + threadIdx.x;
    if (p >= lengths[n])
      continue;

    float3 grid_min = params[n].grid_min;
    float grid_delta = params[n].grid_delta;
    int3 grid_res = params[n].grid_res;

    int3 gc;
    gc.x = (int) ((points[(n*P+p)*3+0]-grid_min.x) * grid_delta);
    gc.y = (int) ((points[(n*P+p)*3+1]-grid_min.y) * grid_delta);
    gc.z = (int) ((points[(n*P+p)*3+2]-grid_min.z) * grid_delta);

    idx_t gs = (gc.x*grid_res.y + gc.y) * grid_res.z + gc.z;
    grid_cell[n*P+p] = gs;
    // for long, need to convert it to unsigned long long since there is no atomicAdd for long
    // grid_idx[n * P + p] = atomicAdd((unsigned long long*)&grid_cnt[n*grid_total + gs], (unsigned long long)1);
    grid_idx[n*P+p] = atomicAdd(&grid_cnt[n*G + gs], 1);
  } 
}

template<typename idx_t>
void InsertPointsCUDA(
    const at::Tensor points,    // (N, P, 3)
    const at::Tensor lengths,   // (N,)
    at::Tensor grid_cnt,        // (N, G)
    at::Tensor grid_cell,       // (N, P)      
    at::Tensor grid_idx,        // (N, P)
    int G,
    const GridParams* params) { // (N,)
  at::TensorArg points_t{points, "points", 1};
  at::TensorArg lengths_t{lengths, "lengths", 2};
  at::TensorArg grid_cnt_t{grid_cnt, "grid_cnt", 3};
  at::TensorArg grid_cell_t{grid_cell, "grid_cell", 4};
  at::TensorArg grid_idx_t{grid_idx, "grid_idx", 5};

  at::CheckedFrom c = "InsertPointsCUDA";
  at::checkAllSameGPU(c, {points_t, lengths_t, grid_cnt_t, grid_cell_t, grid_idx_t});
  at::checkAllSameType(c, {lengths_t, grid_cnt_t, grid_cell_t, grid_idx_t});

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = 256;

  InsertPointsKernel<idx_t><<<blocks, threads, 0, stream>>>(
    points.contiguous().data_ptr<float>(),
    lengths.contiguous().data_ptr<idx_t>(),
    grid_cnt.contiguous().data_ptr<idx_t>(),
    grid_cell.contiguous().data_ptr<idx_t>(),
    grid_idx.contiguous().data_ptr<idx_t>(),
    points.size(0),
    points.size(1),
    G,
    params
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> TestInsertPointsCUDA(
    const at::Tensor bboxes,  
    const at::Tensor points,  
    const at::Tensor lengths,
    float r) {
  int N = bboxes.size(0);
  int P = points.size(1);
  float cell_size = r;
  GridParams* h_params = new GridParams[N];
  int max_grid_total = 0;
  for (size_t i = 0; i < N; ++i) {
    SetupGridParams(
      bboxes.contiguous().data_ptr<float>() + i*6,
      cell_size,
      &h_params[i]
    );
    max_grid_total = std::max(max_grid_total, h_params[i].grid_total);
  }

  GridParams* d_params;
  cudaMalloc((void**)&d_params, N*sizeof(GridParams));
  cudaMemcpy(d_params, h_params, N*sizeof(GridParams), cudaMemcpyHostToDevice);

  auto long_dtype = lengths.options().dtype(at::kLong);
  auto int_dtype = lengths.options().dtype(at::kInt);

  auto dtype = long_dtype;
  dtype = int_dtype;

  auto grid_cnt = at::zeros({N, max_grid_total}, dtype);
  auto grid_cell = at::full({N, P}, -1, dtype); 
  auto grid_idx = at::full({N, P}, -1, dtype);

  InsertPointsCUDA<int>(
    points,
    lengths,
    grid_cnt,
    grid_cell,
    grid_idx,
    max_grid_total,
    d_params
  );

  delete[] h_params;
  cudaFree(d_params);
  return std::make_tuple(grid_cnt, grid_cell, grid_idx);
}
