__global__ void superPixel(int *inputs, float *TOL, int *output)
{
	
	//blockDim.x,y gives the number of threads in a block, in the particular direction
	//gridDim.x,y gives the number of blocks in a grid, in the particular direction
	//blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case)
	
	//block id
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	
	//global thread id
	int globalId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	//local thread id
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;              

	//initialize local sum array to be filled in with values from our input array
	__shared__ int sum[1920/15 * 1080/15];

	//assign values from binary input value array to our local sum array
    sum[localId] = inputs[globalId]; 
    __syncthreads();

    //add up all 1's and 0's in local group using binary reduction
	for (size_t offset = blockDim.x/2; offset > 0 ; offset >>= 1) {
        if (localId < offset) {    
            sum[localId] += sum[localId + offset];
        }
        __syncthreads();
    }
    //ouput final value
    if (localId == 0) {
    	float percentOnes = sum[0]/(blockDim.x*blockDim.y);
	    if (percentOnes > TOL[0]) {
	    	output[blockId] = 1;
	    }
	    else {
	    	output[blockId] = 0;
	    }
	}
}

