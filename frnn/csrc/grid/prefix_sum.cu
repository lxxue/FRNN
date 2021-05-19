// code based on
// https://github.com/ramakarl/fluids3/blob/master/fluids/prefix_sum.cu
// TODO: use template argument to avoid repetition for different data types

// number of shared memory banks is 32 after compute capability 3.5
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index)                                            \
  ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#include "prefix_sum.h"
#include <ATen/ATen.h>
#include <string>

template <bool isNP2>
__device__ void loadSharedChunkFromMemInt(int *s_data, const int *g_idata,
                                          int n, int baseIndex, int &ai,
                                          int &bi, int &mem_ai, int &mem_bi,
                                          int &bankOffsetA, int &bankOffsetB) {
  int thid = threadIdx.x;
  mem_ai = baseIndex + threadIdx.x;
  mem_bi = mem_ai + blockDim.x;

  ai = thid;
  bi = thid + blockDim.x;
  bankOffsetA =
      CONFLICT_FREE_OFFSET(ai); // compute spacing to avoid bank conflicts
  bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  s_data[ai + bankOffsetA] =
      g_idata[mem_ai]; // Cache the computational window in shared memory pad
                       // values beyond n with zeros

  if (isNP2) { // compile-time decision
    s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
  } else {
    s_data[bi + bankOffsetB] = g_idata[mem_bi];
  }
}

template <bool isNP2>
__device__ void storeSharedChunkToMemInt(int *g_odata, const int *s_data, int n,
                                         int ai, int bi, int mem_ai, int mem_bi,
                                         int bankOffsetA, int bankOffsetB) {
  __syncthreads();

  g_odata[mem_ai] = s_data[ai + bankOffsetA]; // write results to global memory
  if (isNP2) {                                // compile-time decision
    if (bi < n)
      g_odata[mem_bi] = s_data[bi + bankOffsetB];
  } else {
    g_odata[mem_bi] = s_data[bi + bankOffsetB];
  }
}

template <bool storeSum>
__device__ void clearLastElementInt(int *s_data, int *g_blockSums,
                                    int blockIndex) {
  if (threadIdx.x == 0) {
    int index = (blockDim.x << 1) - 1;
    index += CONFLICT_FREE_OFFSET(index);
    if (storeSum) { // compile-time decision
      // write this block's total sum to the corresponding index in the
      // blockSums array
      g_blockSums[blockIndex] = s_data[index];
    }
    s_data[index] = 0; // zero the last element in the scan so it will propagate
                       // back to the front
  }
}

__device__ unsigned int buildSumInt(int *s_data) {
  unsigned int thid = threadIdx.x;
  unsigned int stride = 1;

  // build the sum in place up the tree
  for (int d = blockDim.x; d > 0; d >>= 1) {
    __syncthreads();
    if (thid < d) {
      int i = __mul24(__mul24(2, stride), thid);
      int ai = i + stride - 1;
      int bi = ai + stride;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      s_data[bi] += s_data[ai];
    }
    stride *= 2;
  }
  return stride;
}

__device__ void scanRootToLeavesInt(int *s_data, unsigned int stride) {
  unsigned int thid = threadIdx.x;

  // traverse down the tree building the scan in place
  for (int d = 1; d <= blockDim.x; d *= 2) {
    stride >>= 1;
    __syncthreads();

    if (thid < d) {
      int i = __mul24(__mul24(2, stride), thid);
      int ai = i + stride - 1;
      int bi = ai + stride;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      int t = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
}

template <bool storeSum>
__device__ void prescanBlockInt(int *data, int blockIndex, int *blockSums) {
  int stride = buildSumInt(data); // build the sum in place up the tree
  clearLastElementInt<storeSum>(data, blockSums,
                                (blockIndex == 0) ? blockIdx.x : blockIndex);
  scanRootToLeavesInt(data, stride); // traverse down tree to build the scan
}

__global__ void uniformAddInt(int *g_data, int *uniforms, int n,
                              int blockOffset, int baseIndex) {
  __shared__ int uni;
  if (threadIdx.x == 0)
    uni = uniforms[blockIdx.x + blockOffset];
  unsigned int address =
      __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

  __syncthreads();
  // note two adds per thread
  g_data[address] += uni;
  g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

template <bool storeSum, bool isNP2>
__global__ void prescanInt(int *g_odata, const int *g_idata, int *g_blockSums,
                           int n, int blockIndex, int baseIndex) {
  int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
  extern __shared__ int s_dataInt[];
  loadSharedChunkFromMemInt<isNP2>(
      s_dataInt, g_idata, n,
      (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : baseIndex, ai,
      bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
  prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums);
  storeSharedChunkToMemInt<isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi,
                                  bankOffsetA, bankOffsetB);
}

inline bool isPowerOfTwo(int n) { return ((n & (n - 1)) == 0); }

inline int floorPow2(int n) {
  int exp;
  frexp((float)n, &exp);
  return 1 << (exp - 1);
}

#define BLOCK_SIZE 256

int **g_scanBlockSumsInt = 0;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

bool cudaCheck(cudaError_t status, const std::string &msg) {
  if (status != cudaSuccess) {
    printf("CUDA ERROR: %s\n", cudaGetErrorString(status));
    return false;
  } else {
    // app_printf ( "%s. OK.\n", msg );
  }
  return true;
}

void preallocBlockSumsInt(unsigned int maxNumElements) {
  assert(g_numEltsAllocated == 0); // shouldn't be called

  g_numEltsAllocated = maxNumElements;
  unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
  unsigned int numElts = maxNumElements;
  int level = 0;

  do {
    unsigned int numBlocks =
        max(1, (int)ceil((float)numElts / (2.f * blockSize)));
    if (numBlocks > 1)
      level++;
    numElts = numBlocks;
  } while (numElts > 1);

  g_scanBlockSumsInt = (int **)malloc(level * sizeof(int *));
  g_numLevelsAllocated = level;

  numElts = maxNumElements;
  level = 0;

  do {
    unsigned int numBlocks =
        max(1, (int)ceil((float)numElts / (2.f * blockSize)));
    if (numBlocks > 1)
      cudaCheck(cudaMalloc((void **)&g_scanBlockSumsInt[level++],
                           numBlocks * sizeof(int)),
                "Malloc prescanBlockSumsInt g_scanBlockSumsInt");
    numElts = numBlocks;
  } while (numElts > 1);
}

void deallocBlockSumsInt() {
  if (g_scanBlockSumsInt != 0x0) {
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
      cudaCheck(cudaFree(g_scanBlockSumsInt[i]),
                "Malloc deallocBlockSumsInt g_scanBlockSumsInt");
    free((void **)g_scanBlockSumsInt);
  }

  g_scanBlockSumsInt = 0;
  g_numEltsAllocated = 0;
  g_numLevelsAllocated = 0;
}

void prescanArrayRecursiveInt(int *outArray, const int *inArray,
                              int numElements, int level) {
  unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
  unsigned int numBlocks =
      max(1, (int)ceil((float)numElements / (2.f * blockSize)));
  unsigned int numThreads;

  if (numBlocks > 1)
    numThreads = blockSize;
  else if (isPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = floorPow2(numElements);

  unsigned int numEltsPerBlock = numThreads * 2;

  // if this is a non-power-of-2 array, the last block will be non-full
  // compute the smallest power of 2 able to compute its scan.
  unsigned int numEltsLastBlock =
      numElements - (numBlocks - 1) * numEltsPerBlock;
  unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
  unsigned int np2LastBlock = 0;
  unsigned int sharedMemLastBlock = 0;

  if (numEltsLastBlock != numEltsPerBlock) {
    np2LastBlock = 1;
    if (!isPowerOfTwo(numEltsLastBlock))
      numThreadsLastBlock = floorPow2(numEltsLastBlock);
    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
  }

  // padding space is used to avoid shared memory bank conflicts
  unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
  unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
  if (numBlocks > 1)
    assert(g_numEltsAllocated >= numElements);
#endif

  // setup execution parameters
  // if NP2, we process the last block separately
  dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1);
  dim3 threads(numThreads, 1, 1);

  // execute the scan
  if (numBlocks > 1) {
    prescanInt<true, false><<<grid, threads, sharedMemSize>>>(
        outArray, inArray, g_scanBlockSumsInt[level], numThreads * 2, 0, 0);
    if (np2LastBlock) {
      prescanInt<true, true><<<1, numThreadsLastBlock, sharedMemLastBlock>>>(
          outArray, inArray, g_scanBlockSumsInt[level], numEltsLastBlock,
          numBlocks - 1, numElements - numEltsLastBlock);
    }

    // After scanning all the sub-blocks, we are mostly done.  But now we
    // need to take all of the last values of the sub-blocks and scan those.
    // This will give us a new value that must be added to each block to
    // get the final results.
    // recursive (CPU) call
    prescanArrayRecursiveInt(g_scanBlockSumsInt[level],
                             g_scanBlockSumsInt[level], numBlocks, level + 1);

    uniformAddInt<<<grid, threads>>>(outArray, g_scanBlockSumsInt[level],
                                     numElements - numEltsLastBlock, 0, 0);
    if (np2LastBlock) {
      uniformAddInt<<<1, numThreadsLastBlock>>>(
          outArray, g_scanBlockSumsInt[level], numEltsLastBlock, numBlocks - 1,
          numElements - numEltsLastBlock);
    }
  } else if (isPowerOfTwo(numElements)) {
    prescanInt<false, false><<<grid, threads, sharedMemSize>>>(
        outArray, inArray, 0, numThreads * 2, 0, 0);
  } else {
    prescanInt<false, true><<<grid, threads, sharedMemSize>>>(
        outArray, inArray, 0, numElements, 0, 0);
  }
}

// params should be located on cpu
at::Tensor PrefixSumCUDA(const at::Tensor grid_cnt, const at::Tensor params) {
  int N = grid_cnt.size(0);
  int G = grid_cnt.size(1);

  auto params_a = params.accessor<float, 2>();
  // at::Tensor grid_off = at::full({N, G}, -1, grid_cnt.options());
  at::Tensor grid_off = at::full({N, G}, 0, grid_cnt.options());
  for (int n = 0; n < N; ++n) {
    // std::cout << "prefixsum iter " << n << std::endl;
    int num_grids = params_a[n][GRID_3D_TOTAL];
    // std::cout << num_grids << std::endl;

    preallocBlockSumsInt(num_grids);
    prescanArrayRecursiveInt(grid_off.contiguous().data_ptr<int>() + n * G,
                             grid_cnt.contiguous().data_ptr<int>() + n * G,
                             num_grids, 0);
    deallocBlockSumsInt();
  }
  return grid_off;
}