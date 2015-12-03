
__global__ void superPixel(float *inputs, int *output)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int globalId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;              

	// Initialize local sum array to be filled in with values from our input array
	__shared__ float inputsToSum[4];

	// Assign values from input value array to our local sum array
    inputsToSum[localId] = inputs[globalId];
    __syncthreads();

	/////////////////
	// COMPUTATION //
	/////////////////

    if (localId == 0) {
    	for (int i=1; i<4; i++) {
    		inputsToSum[0] = inputsToSum[0] + inputsToSum[i];
    	}
    }
    __syncthreads();

	if (localId == 0) {
    	output[blockId] = inputsToSum[0];
    }

}

