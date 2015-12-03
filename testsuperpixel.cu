
#include <stdio.h>

__global__ void testsuperPixel(int *inputs, int *output)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int globalId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;              
 //    int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
 //    int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
 //    int globalId = (globalIdY * 8) + globalIdX;

	// Initialize local sum array to be filled in with values from our input array
	__shared__ int inputsToSum[16];

	// Assign values from input value array to our local sum array
    inputsToSum[localId] = inputs[globalId];
    __syncthreads();

	/////////////////
	// COMPUTATION //
	/////////////////

    if (localId == 0) {
    	for (int i=1; i<16; i++) { 
    		inputsToSum[0] = inputsToSum[0] + inputsToSum[i];
    	}
    }
    __syncthreads();

    // for (size_t offset = blockDim.x/2; offset > 0 ; offset >>= 1) {
    //     if (localId < offset) {  
    //     	printf("%d",offset);  
    //         inputsToSum[localId] += inputsToSum[localId + offset];
    //     }    
    // }
    // __syncthreads();

	if (globalId == 4) {
        printf("%d\n", threadIdx.x );
        printf("%d\n", threadIdx.y );
        //printf("%d\n", blockDim.y );
        //printf("%d\n", globalId );
        //printf("%d\n", idx );
        printf("%d\n", blockIdx.y);
        printf("%d\n", blockIdx.x);
    	output[blockId] = inputsToSum[0];
    }

}

