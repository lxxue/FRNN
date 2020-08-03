#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "grid.h"

void SetupGridParamsCUDA (
    float* points_min,
    float* points_max,
    float cell_size,
    GridParams* params) {
  std::cout << points_min[0] << ' ' << points_max[0] << std::endl;
  params->grid_min.x = points_min[0];
  params->grid_min.y = points_min[1];
  params->grid_min.z = points_min[2];
  params->grid_max.x = points_max[0];
  params->grid_max.y = points_max[1];
  params->grid_max.z = points_max[2];

  params->grid_size = params->grid_max - params->grid_min;
  params->grid_res.x = (int)(params->grid_size.x / cell_size) + 1;
  params->grid_res.y = (int)(params->grid_size.y / cell_size) + 1;
  params->grid_res.z = (int)(params->grid_size.z / cell_size) + 1;
  params->grid_total = params->grid_res.x * params->grid_res.y * params->grid_res.z;

  params->grid_delta = 1 / cell_size;

  return;
}

void TestSetupGridParamsCUDA (
    at::Tensor bbox_min,
    at::Tensor bbox_max,
    float r) {
  int N = bbox_min.size(0);
  float cell_size = r;
  // std::cout << "cudaMalloc done" << std::endl;
  GridParams* h_params = new GridParams[N];
  for (int i = 0; i < N; ++i) {
    SetupGridParamsCUDA(
      bbox_min.contiguous().data_ptr<float>() + i*3,
      bbox_max.contiguous().data_ptr<float>() + i*3,
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
