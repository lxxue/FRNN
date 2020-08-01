#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "grid.h"
#include "utils/prefix_sum.cuh"

void SetupGridParamsCUDA(
    float* points_max,
    float* points_min,
    float cell_size,
    GridParams& params) {
    // no documentation for at::max so I just do it myself
    std::cout << "setup grid params" << std::endl;

    params.gridMin.x = points_min[0];
    params.gridMin.y = points_min[1];
    params.gridMin.z = points_min[2];
    params.gridMax.x = points_max[0];
    params.gridMax.y = points_max[1];
    params.gridMax.z = points_max[2];
    
    params.gridSize = params.gridMax - params.gridMin;
    params.gridCellSize = cell_size;
    params.gridRes.x = (int)(params.gridSize.x / cell_size) + 1;
    params.gridRes.y = (int)(params.gridSize.y / cell_size) + 1;
    params.gridRes.z = (int)(params.gridSize.z / cell_size) + 1;
    params.gridDelta = 1 / cell_size;
    std::cout << "grid delta done" << std::endl;

    params.gridTotal = params.gridRes.x * params.gridRes.y * params.gridRes.z;
    params.gridSrch = 1;

    std::cout << "grid srch done" << std::endl;
}

__global__ void InsertPointsCUDAKernel(
        const float* __restrict__ Points,
        // int* __restrict__ Grid,
        int* GridCnt,       // not sure if we can use __restrict__ here cause its value would be read
        int* __restrict__ GridCell,
        int* __restrict__ GridIdx,
        int num_points,
        const GridParams* params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    register float3 gridMin = params->gridMin;
    register float gridDelta = params->gridDelta;
    register int3 gridRes = params->gridRes;
    // printf("params set done");
    // register float3 gridMin;
    // gridMin.x = 0.; gridMin.y = 0., gridMin.z = 0.;
    // register float gridDelta = 10.;
    // register int3 gridRes;
    // gridRes.x = 10; gridRes.y = 10; gridRes.z = 10;
    
    register int gs;
    register int3 gc;

    gc.x = (int) ((Points[i*3+0]-gridMin.x) * gridDelta);
    gc.y = (int) ((Points[i*3+1]-gridMin.y) * gridDelta);
    gc.z = (int) ((Points[i*3+2]-gridMin.z) * gridDelta);

    gs = (gc.x*gridRes.y + gc.y) * gridRes.z + gc.z;
    GridCell[i] = gs;
    GridIdx[i] = atomicAdd(&GridCnt[gs], 1);
}

void InsertPointsCUDA(
        const at::Tensor Points,
        at::Tensor Grid,
        at::Tensor GridCnt,
        at::Tensor GridCell,
        at::Tensor GridIdx,
        const GridParams* params) {
    at::TensorArg Points_t{Points, "Points", 1};
    at::TensorArg Grid_t{Grid, "Grid", 2};
    at::TensorArg GridCnt_t{GridCnt, "GridCnt", 3};
    at::TensorArg GridCell_t{GridCell, "GridCell", 4};
    at::TensorArg GridIdx_t{GridIdx, "GridIdx", 5};

    at::CheckedFrom c = "InsertPointsCUDA";
    at::checkAllSameGPU(c, {Points_t, Grid_t, GridCnt_t, GridCell_t, GridIdx_t});
    at::checkAllSameType(c, {Grid_t, GridCnt_t, GridCell_t, GridIdx_t});

    at::cuda::CUDAGuard device_guard(Points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threadsPerBlock = 192;  // Not sure about this value
    int numBlocks = (int)std::ceil((float)Points.size(0) / threadsPerBlock);

    InsertPointsCUDAKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        Points.contiguous().data_ptr<float>(),
        GridCnt.contiguous().data_ptr<int>(),
        GridCell.contiguous().data_ptr<int>(),
        GridIdx.contiguous().data_ptr<int>(),
        Points.size(0),
        params
    );
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: InsertPointsCUDA: %s\n", cudaGetErrorString(error) );
	}  
	cudaDeviceSynchronize();
}

at::Tensor TestGridCUDA(
        const at::Tensor Points,
        const at::Tensor bbox_max,
        const at::Tensor bbox_min,
        int K,
        float r) {
    std::cout << "enter TestGrid" << std::endl;
    float r2 = r * r;
    float cell_size = r;
    GridParams params;
    int num_points = Points.size(0);
    SetupGridParamsCUDA(
        bbox_max.contiguous().data_ptr<float>(),
        bbox_min.contiguous().data_ptr<float>(),
        cell_size, params);
    // copy params to gpu
    // copy params to gpu
    GridParams* d_params;
    // std::cout << "d_params start" << std::endl;
    cudaMalloc((void**)&d_params, sizeof(GridParams));
    // std::cout << "d_params allocated" << std::endl;
    cudaMemcpy(d_params, &params, sizeof(GridParams), cudaMemcpyHostToDevice);
    // std::cout << "d_params copied" << std::endl;
    // std::cout << params.gridMax.x << ' ' << params.gridMax.y << ' ' << params.gridMax.z << std::endl;
    // std::cout << "grid params setup done" << std::endl;

    auto int_dtype = Points.options().dtype(at::kInt);
    
    // not used right now
    at::Tensor Grid = at::full({params.gridTotal}, -1, int_dtype);
    // cell -> #points in this cell
    at::Tensor GridCnt = at::zeros({params.gridTotal}, int_dtype);
    // Point -> cell idx
    at::Tensor GridCell = at::full({num_points, 3}, -1, int_dtype);
    // Point -> next point idx in the same cell
    // at::Tensor GridNext = at::full({num_points}, -1, int_dtype);
    // Point -> idx in its cell
    at::Tensor GridIdx = at::full({num_points}, -1, int_dtype);

    InsertPointsCUDA(Points, Grid, GridCnt, GridCell, GridIdx, d_params);
    std::cout << "points inserted" << std::endl;

    return GridCnt;
}

at::Tensor PrefixSum(at::Tensor GridCnt) {
    int num_grids = GridCnt.size(0);
    at::Tensor GridOff = at::zeros({num_grids}, GridCnt.options());
    preallocBlockSumsInt(num_grids);
    prescanArrayRecursiveInt(
        GridOff.contiguous().data_ptr<int>(),
        GridCnt.contiguous().data_ptr<int>(),
        num_grids,
        0
    );
    cudaDeviceSynchronize();
    deallocBlockSumsInt();
    return GridOff;
}