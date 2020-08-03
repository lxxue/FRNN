#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "grid.h"

void SetupGridParamsCUDA (
    float* bboxes,
    float cell_size,
    GridParams* params) {
  std::cout << bboxes[0] << ' ' << bboxes[3] << std::endl;
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

void TestSetupGridParamsCUDA (
    at::Tensor bboxes,  // N x 3 x 2 (min, max) at last dimension
    float r) {
  int N = bboxes.size(0);
  // TODO: cell_size determined joint by search radius and bbox_size
  // TODO: cell_size different for different point clouds in the batch
  float cell_size = r;
  // std::cout << "cudaMalloc done" << std::endl;
  GridParams* h_params = new GridParams[N];
  for (int i = 0; i < N; ++i) {
    SetupGridParamsCUDA(
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
    size_t P,
    const GridParams* params) {


}

