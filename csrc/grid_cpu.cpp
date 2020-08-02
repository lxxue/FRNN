#include <torch/extension.h>
#include <iostream>
#include <queue>
#include "grid.h"
#include "grid.cuh"

void SetupGridParams(
    const at::Tensor Points,
    float cell_size,
    GridParams& params) {
    // no documentation for at::max so I just do it myself
    std::cout << "setup grid params" << std::endl;
    int num_points = Points.size(0);
    auto Points_a = Points.accessor<float, 2>();
    // assume there is at least one point in the cloud
    params.gridMin.x = params.gridMax.x = Points_a[0][0];
    params.gridMin.y = params.gridMax.y = Points_a[0][1];
    params.gridMin.z = params.gridMax.z = Points_a[0][2];
    std::cout << "grid for loop starts" << std::endl;
    for (int i = 1; i < num_points; ++i) {
        params.gridMin.x = std::min(params.gridMin.x, Points_a[i][0]);
        params.gridMax.x = std::max(params.gridMax.x, Points_a[i][0]);
        params.gridMin.y = std::min(params.gridMin.y, Points_a[i][1]);
        params.gridMax.y = std::max(params.gridMax.y, Points_a[i][1]);
        params.gridMin.z = std::min(params.gridMin.z, Points_a[i][2]);
        params.gridMax.z = std::max(params.gridMax.z, Points_a[i][2]);
    }
    std::cout << "grid min max done" << std::endl;
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

    // params.gridSrch = std::floor(2*search_radius/params.gridCellSize) + 1;
    // if (params.gridSrch < 3) params.gridSrch = 3; 
    // assert(params.gridSrch < 6);
    // params.gridAdjCnt = params.gridSrch * params.gridSrch * params.gridSrch;
    // int cell = 0;
    // for (int x=0; x < params.gridSrch; x++) {
    //     for (int y=0; y < params.gridSrch; y++) {
    //         for (int z=0; z < params.gridSrch; z++) {
    //             params.gridAdj[cell++] = (x*params.gridRes.y + y)*params.gridRes.z + z;
    //         }
    //     }
    // }
}

int getGridCell(float x, float y, float z, int3& gc, GridParams& params) {
    gc.x = (int) ((x - params.gridMin.x) * params.gridDelta);
    gc.y = (int) ((y - params.gridMin.y) * params.gridDelta);
    gc.z = (int) ((z - params.gridMin.z) * params.gridDelta);

    return (gc.x*params.gridRes.y + gc.y)*params.gridRes.z + gc.z;
}

void InsertPoints(const at::Tensor Points, at::Tensor Grid, at::Tensor GridCnt, 
                  at::Tensor GridCell, at::Tensor GridNext, GridParams& params) {
    auto Points_a = Points.accessor<float, 2>();
    auto Grid_a = Grid.accessor<int, 3>();
    auto GridCnt_a = GridCnt.accessor<int, 3>();
    auto GridCell_a = GridCell.accessor<int, 2>();
    auto GridNext_a = GridNext.accessor<int, 1>();

    int gs;
    int3 gc;
    for (int n=0; n < Points.size(0); n++) {
        gs = getGridCell(Points_a[n][0], Points_a[n][1], Points_a[n][2], gc, params);
        // std::cout << Points_a[n][0] << ' ' << Points_a[n][1] << ' ' << Points_a[n][2] << std::endl;
        // std::cout << gc.x << ' ' << gc.y << ' ' << gc.z << std::endl;
        // if (gc.x >= params.gridSrch/2 && gc.x < (params.gridRes.x-params.gridSrch/2) &&
        //     gc.y >= params.gridSrch/2 && gc.y < (params.gridRes.y-params.gridSrch/2) &&
        //     gc.z >= params.gridSrch/2 && gc.z < (params.gridRes.z-params.gridSrch/2)) {
        GridCell_a[n][0] = gc.x; GridCell_a[n][1] = gc.y; GridCell_a[n][2] = gc.z;
        GridNext_a[n] = Grid_a[gc.x][gc.y][gc.z];
        Grid_a[gc.x][gc.y][gc.z] = n;
        GridCnt_a[gc.x][gc.y][gc.z]++;
        // }
    }
}

std::tuple<at::Tensor, at::Tensor> FindNbrsGrid(
    const at::Tensor Points, 
    const at::Tensor Grid,
    const at::Tensor GridNext, 
    const at::Tensor GridCell,
    const GridParams& params,
    int K,
    float r2) {

    auto Points_a = Points.accessor<float, 2>();
    auto Grid_a = Grid.accessor<int, 3>();
    auto GridCell_a = GridCell.accessor<int, 2>();
    auto GridNext_a = GridNext.accessor<int, 1>();

    float3 dst;
    float dsq;
    // int nadj = (params.gridRes.y+1)*params.gridRes.z + 1
    int num_points = Points.size(0);

    auto long_opts = Points.options().dtype(torch::kInt64);
    torch::Tensor idxs = torch::full({num_points, K}, -1, long_opts);
    torch::Tensor dists = torch::full({num_points, K}, -1, Points.options());

    auto idxs_a = idxs.accessor<long, 2>();
    auto dists_a = dists.accessor<float, 2>();
    

    for (int i=0; i < num_points; ++i) {
        std::priority_queue<std::tuple<float, int>> q;
        float px = Points_a[i][0], py = Points_a[i][1], pz = Points_a[i][2];
        int cx = GridCell_a[i][0], cy = GridCell_a[i][1], cz = GridCell_a[i][2];
        int startx = std::max(0, cx-params.gridSrch), endx = std::min(cx+params.gridSrch, params.gridRes.x-1);
        int starty = std::max(0, cy-params.gridSrch), endy = std::min(cy+params.gridSrch, params.gridRes.y-1);
        int startz = std::max(0, cz-params.gridSrch), endz = std::min(cz+params.gridSrch, params.gridRes.z-1);
        // std::cout << px << ' ' << py << ' ' << pz << std::endl;
        // std::cout << cx << ' ' << cy << ' ' << cz << std::endl;
        
        // std::cout << startx << ' ' << endx << std::endl;
        // std::cout << starty << ' ' << endy << std::endl;
        // std::cout << startz << ' ' << endz << std::endl;
        for (int x=startx; x<=endx; ++x) {
            for (int y=starty; y<=endy; ++y) {
                for (int z=startz; z<=endz; ++z) {
                    int cur = Grid_a[x][y][z];
                    while (cur != -1) {
                        // std::cout << cur << std::endl;
                        // for testing against benchmarks
                        if (cur != i || true) {
                            dst.x = Points_a[cur][0] - px, dst.y = Points_a[cur][1] - py, dst.z = Points_a[cur][2] - pz;
                            dsq = dst.x*dst.x + dst.y*dst.y + dst.z*dst.z;
                            if (dsq <= r2) {
                                int size = static_cast<int>(q.size());
                                assert(q.size() <= K);
                                if (size < K && dsq < r2) {
                                    q.emplace(dsq, cur);
                                }
                                else if (size == K && dsq < r2 && dsq < std::get<0>(q.top())) {
                                    q.emplace(dsq, cur);
                                    q.pop();
                                }
                            }
                        }
                        cur = GridNext_a[cur]; 
                    }                    
                }
            }
        }
        while (!q.empty()) {
            auto t = q.top();
            q.pop();
            const int k = q.size();
            dists_a[i][k] = std::get<0>(t);
            idxs_a[i][k] = std::get<1>(t);
        }
    }
    return std::make_tuple(idxs, dists);
}

std::tuple<at::Tensor, at::Tensor> TestGrid(
    const at::Tensor Points, int K, float r) {
    std::cout << "enter TestGrid" << std::endl;
    float r2 = r * r;
    // The ideal grid to search on one dimension should be 2~4. We choose 3 here. 
    float cell_size = r;
    GridParams params;
    int num_points = Points.size(0);
    SetupGridParams(Points, cell_size, params);
    std::cout << "grid params setup done" << std::endl;

    // uniform grid data structure
    // last point idx in this grid cell
    at::Tensor Grid = at::full({params.gridRes.x, params.gridRes.y, params.gridRes.z}, -1, at::kInt);
    at::Tensor GridCnt = at::zeros({params.gridRes.x, params.gridRes.y, params.gridRes.z}, at::kInt);
    // Point -> cell idx
    at::Tensor GridCell = at::full({num_points, 3}, -1, at::kInt);
    // Point -> next point idx in the same cell
    at::Tensor GridNext = at::full({num_points}, -1, at::kInt);

    // results: current only one pointcloud is supported 

    InsertPoints(Points, Grid, GridCnt, GridCell, GridNext, params);
    std::cout << "points inserted" << std::endl;

    // auto GridCnt_a = GridCnt.accessor<int, 3>();
    // for (int x=1; x<params.gridRes.x-1; ++x) {
    //     for (int y=1; y<params.gridRes.y-1; ++y) {
    //         for (int z=1; z<params.gridRes.z-1; ++z) {
    //             if (GridCnt_a[x][y][z] > 0) {
    //                 std::cout << x << ' ' << y << ' ' << z << std::endl;
    //                 std::cout << GridCnt_a[x][y][z] << std::endl; 
    //             }
    //         }
    //     }
    // }
    
    
    return FindNbrsGrid(Points, Grid, GridNext, GridCell, params, K, r2);
    // return GridCnt;
}
