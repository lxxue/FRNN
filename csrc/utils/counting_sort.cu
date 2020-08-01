
// Counting Sort - Index
__global__ void countingSortIndex (int* GridCell, int* GridIdx, int* GridOff, int* SortedPointIdxs, int num_points)
{
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= num_points ) return;

	int icell = GridCell[i];
	int indx = GridIdx[i];
	int sort_ndx = Gridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
    SortedPointIdxs[ sort_ndx ] = i;					// index sort, grid refers to original particle order
}

// Counting Sort - Full (deep copy)
__global__ void countingSortFull (int* GridCell, int* GridIdx, int* GridOff, float* Points, int* SortedGridCell, float* SortedPoints, int num_points)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= num_points ) return;

	// Copy points from original, unsorted Points,
	// into sorted memory location on device (mpos/mvel)
	int icell = GridCell[i];
	int indx = GridIdx[i];		

	// Determine the sort_ndx, location of the particle after sort
	int sort_ndx = GridOff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset	
		
    // Find the original particle data, offset into unsorted buffer (msortbuf)
	// char* bpos = buf.msortbuf + i*sizeof(float3);

	// Transfer data to sort location
    // buf.mgrid[ sort_ndx ] = sort_ndx;			// full sort, grid indexing becomes identity		
    SortedPoints[sort_ndx*3] = Points[i*3];
    SortedPoints[sort_ndx*3+1] = Points[i*3+1];
    SortedPoints[sort_ndx*3+2] = Points[i*3+2];

    SortedGridCell[sort_ndx] = icell;
    // no need for idx after sorting?
    // GridIdx[sort_ndx] = indx;
}

void CountingSortIndexCUDA (at::Tensor GridCell, 
                            at::Tensor GridIdx, 
                            at::Tensor GridOff, 
                            at::Tensor SortedPointIdxs)
{	
    int threadsPerBlock = 192;  // Not sure about this value
    int numBlocks = (int)std::ceil((float)Points.size(0) / threadsPerBlock);
	countingSortIndex <<< numBlocks, threadsPerBlock>>> (
        GridCell.contiguous().data_ptr<int>(),
        GridIdx.contiguous().data_ptr<int>(),
        GridOff.contiguous().data_ptr<int>(),
        SortedPointIdxs.contiguous().data_ptr<int>(),
        GridCell.size(0);
    );		
	cudaThreadSynchronize ();
}

void CountingSortIndexCUDA (at::Tensor GridCell, 
                            at::Tensor GridIdx, 
                            at::Tensor GridOff, 
                            at::Tensor Points,
                            at::Tensor SortedGridCell,
                            at::Tensor SortedPoints)
{
    int threadsPerBlock = 192;  // Not sure about this value
    int numBlocks = (int)std::ceil((float)Points.size(0) / threadsPerBlock);
	countingSortFull <<< numBlocks, threadsPerBlock>>> (
        GridCell.contiguous().data_ptr<int>(),
        GridIdx.contiguous().data_ptr<int>(),
        GridOff.contiguous().data_ptr<int>(),
        Points.contiguous().data_ptr<float>(),
        SortedGridCell.contiguous().data_ptr<int>(),
        SortedPoints.contiguous().data_ptr<float>(),
        GridCell.size(0);
    );		
	cudaThreadSynchronize ();
}