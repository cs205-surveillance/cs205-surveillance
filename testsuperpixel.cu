
#include <stdio.h>

__global__ void testsuperPixel(int *inputs, int *output)
{
	
    int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalId = (globalIdY * 4) + globalIdX;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	//int globalId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;              
    
    __shared__ int inputsToSum[8];
    inputsToSum[localId] = inputs[globalId];

    if (blockId == 2) {
        printf("%d",inputsToSum[localId]);

    }

    // if (localId == 0) {
    //  for (int i=1; i<4; i++) { 
    //      inputsToSum[0] = inputsToSum[0] + inputsToSum[i];
    //  }
    // }
    // __syncthreads();

 //    // int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
 //    // int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
 //    // int globalId = (globalIdY * 4) + globalIdX;

	// // Initialize local sum array to be filled in with values from our input array
	// __shared__ int inputsToSum[8];

	// // Assign values from input value array to our local sum array
 //    inputsToSum[localId] = inputs[globalId];
 //    __syncthreads();

	// /////////////////
	// // COMPUTATION //
	// /////////////////

 //    if (localId == 0) {
 //    	for (int i=1; i<4; i++) { 
 //    		inputsToSum[0] = inputsToSum[0] + inputsToSum[i];
 //    	}
 //    }
 //    __syncthreads();

 //    // for (size_t offset = blockDim.x/2; offset > 0 ; offset >>= 1) {
 //    //     if (localId < offset) {  
 //    //     	printf("%d",offset);  
 //    //         inputsToSum[localId] += inputsToSum[localId + offset];
 //    //     }    
 //    // }
 //    // __syncthreads();
 //    printf("%d\n", blockId);
	// if (localId == 0) {
 //        printf("%d\n", threadIdx.x );
 //        printf("%d\n", threadIdx.y );
 //        //printf("%d\n", blockDim.y );
 //        //printf("%d\n", globalId );
 //        //printf("%d\n", idx );
 //        printf("%d\n", blockIdx.y);
 //        printf("%d\n", blockIdx.x);
 //    	output[blockId] = inputsToSum[0];
  //  }

}

