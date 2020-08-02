#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "grid.h"
#include "utils/prefix_sum.cuh"
#include "utils/counting_sort.cuh"
#include "utils/mink.cuh"



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

    std::cout << params.gridMin.x << ' ' << params.gridMin.y << ' ' << params.gridMin.z << std::endl;
    std::cout << params.gridMax.x << ' ' << params.gridMax.y << ' ' << params.gridMax.z << std::endl;
    
    params.gridSize = params.gridMax - params.gridMin;
    params.gridCellSize = cell_size;
    params.gridRes.x = (int)(params.gridSize.x / cell_size) + 1;
    params.gridRes.y = (int)(params.gridSize.y / cell_size) + 1;
    params.gridRes.z = (int)(params.gridSize.z / cell_size) + 1;
    params.gridDelta = 1 / cell_size;
    std::cout << "grid delta done" << std::endl;

    params.gridTotal = params.gridRes.x * params.gridRes.y * params.gridRes.z;
    params.gridSrch = 2;

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

std::tuple<at::Tensor, at::Tensor> TestGridCUDA(
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
    at::Tensor GridCell = at::full({num_points}, -1, int_dtype);
    // Point -> next point idx in the same cell
    // at::Tensor GridNext = at::full({num_points}, -1, int_dtype);
    // Point -> idx in its cell
    at::Tensor GridIdx = at::full({num_points}, -1, int_dtype);
    
    // new Points and GridCell after sorting
    at::Tensor SortedGridCell = at::zeros({num_points}, GridCell.options());
    at::Tensor SortedPoints = at::zeros({num_points, 3}, Points.options());
    at::Tensor SortedIdx = at::zeros({num_points}, GridCell.options());

    InsertPointsCUDA(Points, Grid, GridCnt, GridCell, GridIdx, d_params);
    std::cout << "points inserted" << std::endl;
    // return std::make_tuple(GridCnt, GridCell);

    at::Tensor GridOff = PrefixSum(GridCnt);
    CountingSortFullCUDA (
        GridCell, 
        GridIdx, 
        GridOff, 
        Points,
        SortedGridCell,
        SortedPoints,
        SortedIdx
    );

    return FindNbrsGridCUDA(SortedPoints, GridOff, SortedGridCell, SortedIdx, d_params, K, r2);
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

// fix K to be 5 for now
// template later
__global__ void FindNbrsGridKernel(
    const float* __restrict__ Points, 
    const int* __restrict__ GridOff,
    const int* __restrict__ GridCell,
    const int* __restrict__ SortedIdx,
    float* __restrict__ dists,
    long* __restrict__ idxs,
    const GridParams* params,
    int num_points,
    float r2) {

    const int K = 5;

    int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num_points) return;
    float3 dist;
    float dsq;

    register float px = Points[i*3], py = Points[i*3+1], pz = Points[i*3+2];
    register int res_x = params->gridRes.x, res_y = params->gridRes.y, res_z = params->gridRes.z;
    register int grid_srch = params->gridSrch;
    register int grid_total = params->gridTotal;
    int grid_idx = GridCell[i];
    // gs = gc.x*params.gridRes.y + gc.y)*params.gridRes.z + gc.z
    int cz = grid_idx % res_z;
    int cy = (grid_idx / res_z) % res_y;
    int cx = (grid_idx / res_z) / res_y;
    // printf("%f %f %f %d %d %d\n", px, py, pz, cx, cy, cz);
    int startx = std::max(0, cx-grid_srch), endx = std::min(cx+grid_srch, res_x-1);
    int starty = std::max(0, cy-grid_srch), endy = std::min(cy+grid_srch, res_y-1);
    int startz = std::max(0, cz-grid_srch), endz = std::min(cz+grid_srch, res_z-1);

    int original_i = SortedIdx[i];
    if (original_i == 1) {
        printf("%d %d %d %d %d %d %d\n", i, startx, endx, starty, endy, startz, endz);
    }

    float min_dists[5];
    int min_idxs[5];
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x=startx; x<=endx; ++x) {
        for (int y=starty; y<=endy; ++y) {
            for (int z=startz; z<=endz; ++z) {
                int cur = (x*res_y + y)*res_z + z;
                // int p_start = ((cur-1) >= 0 ? GridOff[cur-1] : 0);
                // int p_end = GridOff[cur];
                int p_start = GridOff[cur];
                int p_end = cur+1 == grid_total ? num_points : GridOff[cur+1];

                // if (original_i == 1) {
                //     printf("%d %d %d %d %d %d\n", x, y, z, cur, p_start, p_end);
                // }

                for (int p=p_start; p < p_end; ++p) {
                    dist.x = Points[p*3] - px;
                    dist.y = Points[p*3+1] - py;
                    dist.z = Points[p*3+2] - pz;
                    dsq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
                    if (dsq <= r2) {
                        // printf("%f %f %f %f %f %f %d %d %f\n", px, py, pz, Points[p*3], Points[p*3+1], Points[p*3+2], i, p, dsq);
                        mink.add(dsq, SortedIdx[p]);
                    }
                }
                mink.sort();
                for (int k=0; k < mink.size(); ++k) {
                    idxs[original_i*K+k] = min_idxs[k];
                    dists[original_i*K+k] = min_dists[k];
                }
            }
        }
    }
}

std::tuple<at::Tensor, at::Tensor> FindNbrsGridCUDA(
    at::Tensor Points,
    at::Tensor GridOff,
    at::Tensor GridCell,
    at::Tensor SortedIdx,
    const GridParams* params,
    int K,
    float r2) 
{
    int threadsPerBlock = 192;  // Not sure about this value
    int numBlocks = (int)std::ceil((float)Points.size(0) / threadsPerBlock);
    int num_points = Points.size(0);

    at::TensorArg Points_t{Points, "Points", 1};
    at::TensorArg GridOff_t{GridOff, "GridOff", 2};
    at::TensorArg GridCell_t{GridCell, "GridCell", 3};

    at::CheckedFrom c = "FindNbrsGridCUDA";
    at::checkAllSameGPU(c, {Points_t, GridOff_t, GridCell_t});
    at::checkAllSameType(c, {GridOff_t, GridCell_t});

    at::cuda::CUDAGuard device_guard(Points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();



    auto long_opts = Points.options().dtype(torch::kInt64);
    torch::Tensor idxs = torch::full({num_points, K}, -1, long_opts);
    torch::Tensor dists = torch::full({num_points, K}, -1, Points.options());
    
    FindNbrsGridKernel <<< numBlocks, threadsPerBlock, 0, stream >>> (
        Points.contiguous().data_ptr<float>(),
        GridOff.contiguous().data_ptr<int>(),
        GridCell.contiguous().data_ptr<int>(),
        SortedIdx.contiguous().data_ptr<int>(),
        dists.contiguous().data_ptr<float>(),
        idxs.contiguous().data_ptr<long>(),
        params,
        num_points,
        r2
    );
    return std::make_tuple(idxs, dists);
}

/*
__global__ void FindNbrsGridNoSortingKernel(
    const float* __restrict__ Points, 
    const int* __restrict__ Grid,
    const int* __restrict__ GridNext,
    const int* __restrict__ GridCell,
    float* __restrict__ dists,
    long* __restrict__ idxs,
    const GridParams* params,
    int num_points,
    float r2) {

    const int K = 5;

    int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num_points) return;

    float3 dist;
    float dsq;

    register float px = Points[i*3], py = Points[i*3+1], pz = Points[i*3+2];
    register int res_x = params->gridRes.x, res_y = params->gridRes.y, res_z = params->gridRes.z;
    register int grid_srch = params->gridSrch;
    int grid_idx = GridCell[i];
    // gs = gc.x*params.gridRes.y + gc.y)*params.gridRes.z + gc.z
    int cz = grid_idx % res_z;
    int cy = (grid_idx / res_z) % res_y;
    int cx = (grid_idx / res_z) / res_y;
    // printf("%f %f %f %d %d %d\n", px, py, pz, cx, cy, cz);
    int startx = std::max(0, cx-grid_srch), endx = std::min(cx+grid_srch, res_x-1);
    int starty = std::max(0, cy-grid_srch), endy = std::min(cy+grid_srch, res_y-1);
    int startz = std::max(0, cz-grid_srch), endz = std::min(cz+grid_srch, res_z-1);

    float min_dists[5];
    int min_idxs[5];
    MinK<float, int> mink(min_dists, min_idxs, K);
    for (int x=startx; x<=endx; ++x) {
        for (int y=starty; y<=endy; ++y) {
            for (int z=startz; z<=endz; ++z) {
                int grid_idx = (x*res_y + y)*res_z + z
                int cur = Grid[grid_idx] 
                while (cur != -1) {
                    if (cur != i || true) {
                        dist.x = Points[cur*3] - px;
                        dist.y = Points[cur*3+1] - py;
                        dist.z = Points[cur*3+2] - pz;
                        dsq = dist.x*dist.x + dist.y*dist.y + dist.z*dist.z;
                        if (dsp <= r2) {
                            mink.add(dsq, cur);
                        }
                    }
                    cur = GridNext[cur];
                }
                mink.sort();
                for (int k = 0; k < mink.size(); ++k) {
                    idxs[i*K+k] = min_idxs[k];
                    dists[i*K+k] = min_dists[k];
                }
            }
        }
    }
}


std::tuple<at::Tensor, at::Tensor> FindNbrsGridNoSortingCUDA(
    at::Tensor Points,
    at::Tensor Grid,
    at::Tensor GridCell,
    at::Tensor GridNext,
    const GridParams* params,
    int K,
    float r2) 
{
    int threadsPerBlock = 192;  // Not sure about this value
    int numBlocks = (int)std::ceil((float)Points.size(0) / threadsPerBlock);
    int num_points = Points.size(0);

    at::TensorArg Points_t{Points, "Points", 1};
    at::TensorArg Grid_t{Grid, "GridOff", 2};
    at::TensorArg GridCell_t{GridCell, "GridCell", 3};
    at::TensorArg GridNext_t{GridNext, "GridCell", 3};

    at::CheckedFrom c = "FindNbrsGridNoSortingCUDA";
    at::checkAllSameGPU(c, {Points_t, Grid_t, GridCell_t, GridNext_t});
    at::checkAllSameType(c, {Grid_t, GridCell_t, GridNext_t});

    at::cuda::CUDAGuard device_guard(Points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto long_opts = Points.options().dtype(torch::kInt64);
    torch::Tensor idxs = torch::full({num_points, K}, -1, long_opts);
    torch::Tensor dists = torch::full({num_points, K}, -1, Points.options());
    
    FindNbrsGridNoSortingKernel <<< numBlocks, threadsPerBlock, 0, stream >>> (
        Points.contiguous().data_ptr<float>(),
        Grid.contiguous().data_ptr<int>(),
        GridCell.contiguous().data_ptr<int>(),
        GridNext.contiguous().data_ptr<int>(),
        dists.contiguous().data_ptr<float>(),
        idxs.contiguous().data_ptr<long>(),
        params,
        num_points,
        r2
    );
    return std::make_tuple(idxs, dists);
}


__global__ void InsertPointsNoSortingKernel(
        const float* __restrict__ Points,
        int* Grid,
        int* GridCnt,       // not sure if we can use __restrict__ here cause its value would be read
        int* __restrict__ GridCell,
        int* __restrict__ GridNext,
        int num_points,
        const GridParams* params) {

    int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num_points) return;

    register float3 gridMin = params->gridMin;
    register float gridDelta = params->gridDelta;
    register int3 gridRes = params->gridRes;
    
    register int gs;
    register int3 gc;

    gc.x = (int) ((Points[i*3+0]-gridMin.x) * gridDelta);
    gc.y = (int) ((Points[i*3+1]-gridMin.y) * gridDelta);
    gc.z = (int) ((Points[i*3+2]-gridMin.z) * gridDelta);

    // Problem here: I don't know how to parallel here
    gs = (gc.x*gridRes.y + gc.y) * gridRes.z + gc.z;
    GridCell[i] = gs;
    GridNext[i] = Grid[gs];
    Grid[gs] = i;
    atomicAdd(&GridCnt[gs], 1);
}

std::tuple<at::Tensor, at::Tensor> TestGridNoSortingCUDA(
        const at::Tensor Points,
        const at::Tensor bbox_max,
        const at::Tensor bbox_min,
        int K,
        float r) {
    std::cout << "enter TestGridNoSortingCUDA" << std::endl;
    float r2 = r * r;
    float cell_size = r;
    GridParams params;
    int num_points = Points.size(0);
    SetupGridParamsCUDA(
        bbox_max.contiguous().data_ptr<float>(),
        bbox_min.contiguous().data_ptr<float>(),
        cell_size, 
        params);
    GridParams* d_params;
    cudaMalloc((void**)&d_params, sizeof(GridParams));
    cudaMemcpy(d_params, &params, sizeof(GridParams), cudaMemcpyHostToDevice);

    auto int_dtype = Points.options().dtype(at::kInt);
    
    // cell -> first point's idx in this cell
    at::Tensor Grid = at::full({params.gridTotal}, -1, int_dtype);
    // cell -> #points in this cell
    at::Tensor GridCnt = at::zeros({params.gridTotal}, int_dtype);
    // Point -> cell idx
    at::Tensor GridCell = at::full({num_points, 3}, -1, int_dtype);
    // Point -> next point idx in the same cell
    at::Tensor GridNext = at::full({num_points}, -1, int_dtype);
    // Point -> idx in its cell
    // at::Tensor GridIdx = at::full({num_points}, -1, int_dtype);
    
    // new Points and GridCell after sorting
    // at::Tensor SortedGridCell = at::zeros({num_points}, GridCell.options());
    // at::Tensor SortedPoints = at::zeros({num_points, 3}, Points.options());

    InsertPointsCUDA(Points, Grid, GridCnt, GridCell, GridIdx, d_params);
    std::cout << "points inserted" << std::endl;

    at::Tensor GridOff = PrefixSum(GridCnt);
    CountingSortFullCUDA (
        GridCell, 
        GridIdx, 
        GridOff, 
        Points,
        SortedGridCell,
        SortedPoints
    );

    return FindNbrsGridCUDA(SortedPoints, GridOff, SortedGridCell, d_params, K, r2);
}
*/
