#include <stdio.h>
#include <string>

// number of shared memory banks is 32 after compute capability 3.5
#define NUM_BANKS		32
#define LOG_NUM_BANKS	 5

void preallocBlockSumsInt (unsigned int maxNumElements);
void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level);
void deallocBlockSumsInt();

bool cudaCheck ( cudaError_t status, const std::string& msg );