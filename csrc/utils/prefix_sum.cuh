#include <stdio.h>
#include <string>


#define NUM_BANKS		16
#define LOG_NUM_BANKS	 4

void preallocBlockSumsInt (unsigned int maxNumElements);
void prescanArrayRecursiveInt (int *outArray, const int *inArray, int numElements, int level);
void deallocBlockSumsInt();

bool cudaCheck ( cudaError_t status, const std::string& msg );