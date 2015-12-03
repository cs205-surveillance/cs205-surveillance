__global__ void superPixel(int *inputs, float *TOL, int *output)
{
	////////////////////
	// KERNEL OVERVIEW /
	////////////////////

	/*
	This kernel is used to determine if there is an anomaly in a superpixel.
	This kernel takes in a 1080x1920 continuous array and calculates the sum for each
	30x30 pixel block (i.e., superpixel). Once the summation for every block is 
	complete, the kernel calculates the sum over area for the whole superpixel.
	If that value is above our tolerance, the kernel returns a 1 for that superpixel.
	Else, it returns a 0. The final output is a 1D array, where each value in
	that array corresponds to one of the superpixels.
	*/ 

	//////////////////////////
	// INITIALIZE PARAMETERS /
	//////////////////////////

	//blockDim.x gives the number of threads in a block (x direction)
	//gridDim.x gives the number of blocks in a grid (x direction)
	//blockDim.x * gridDim.x gives the number of threads in a grid (x direction)
	
	//block id
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	
	//global thread id
	int globalId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	//local thread id
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;              

	//initialize local sum array to be filled in with values from our input array
	__shared__ int sum[1920/blockDim.y * 1080/blockDim.x];

	//assign values from input value array to our local sum array
    sum[localId] = inputs[globalId]; 
    __syncthreads();

	////////////////
	// COMPUTATION /
	////////////////

    //add up all values in local group using binary reduction
	for (size_t offset = blockDim.x/2; offset > 0 ; offset >>= 1) {
        if (localId < offset) {    
            sum[localId] += sum[localId + offset];
        }
        __syncthreads();
    }
    
    //ouput final value
    if (localId == 0) {
    	float fraction = sum[0]/(blockDim.x*blockDim.y);
	    if (fraction > TOL[0]) {
	    	output[blockId] = 1;
	    }
	    else {
	    	output[blockId] = 0;
	    }
	}
}

