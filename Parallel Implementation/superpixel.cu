__global__ void superPixel(float *inputs, int *output)
{
	////////////////////
	// KERNEL OVERVIEW /
	////////////////////

	/*
	This kernel is used to determine if there is an anomaly in a superpixel.
	This kernel takes in a 1080x1920 array (the output from minimum filter kernel) 
	and calculates the sum for each 30x30 pixel block (i.e., superpixel). Once the 
	summation for every block is complete, the kernel determines if the sum is above our
	tolerance. If so, the kernel returns a 1 for that superpixel. Else, it returns a 0. The 
	final output is a 1D array, where each value in that array corresponds to one of the superpixels.
	*/ 

	// blockDim.x gives the number of threads in a block (x direction)
	// gridDim.x gives the number of blocks in a grid (x direction)
	// blockDim.x * gridDim.x gives the number of threads in a grid (x direction)
	
	// Initialize parameters
    int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalId = (globalIdY * 1920) + globalIdX;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;               

	// Initialize local array to be filled with values from input array
	__shared__ float inputsToSum[30*30];

	// Assign values from input value array to local sum array
    inputsToSum[localId] = inputs[globalId];
    __syncthreads();

    // First thread in each row will compute a row sum
	if (localId % 30 == 0) {
    	for (int i = localId + 1; i < localId + 30; i++) {
    		inputsToSum[localId] += inputsToSum[i];
    	}
    }
	__syncthreads();

	// One thread will combine all above sums to get single value
	if (localId == 0) {
    	for (int i=0; i<900; i+= 30) {
    		inputsToSum[0] += inputsToSum[i];
    	}
    }

    // Assign 1 if sum above threshold. 
    if (localId == 0) {
        if (inputsToSum[0] > 15*700) { 
            output[blockId] = 1;
        }  
        else{
        	output[blockId] = 0;
        }
    }
}

