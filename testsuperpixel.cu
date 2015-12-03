
#include <stdio.h>

__global__ void testsuperPixel(int *inputs, int *output)
{
	//these index expressions seemingly work
    int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalId = (globalIdY * 10) + globalIdX;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;              
    
    __shared__ int inputsToSum[4];
    inputsToSum[localId] = inputs[globalId];
    
    if (globalId == 12) {
        for (int i=0; i<4; i++) { 
            printf("%d\n",inputsToSum[i]);
            __syncthreads();
        }
    }

    //this sums up each block/superpixel
    if (localId == 0) {
        for (int i=1; i<4; i++) { 
            inputsToSum[0] = inputsToSum[0] + inputsToSum[i];
            __syncthreads();
        }
        
    }
    __syncthreads();
    if (localId == 0) {
        if (inputsToSum[0] > 1) { 
            output[blockId] = inputsToSum[0];
        }  
    }
           

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

