
#include <stdio.h>

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

	///////////////////////////
	// INITIALIZE PARAMETERS //
	///////////////////////////

	// blockDim.x gives the number of threads in a block (x direction)
	// gridDim.x gives the number of blocks in a grid (x direction)
	// blockDim.x * gridDim.x gives the number of threads in a grid (x direction)

    int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalId = (globalIdY * 1920) + globalIdX;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;

    // Virtual 32 x 32
	int yStart = 32 * blockIdx.y;
	int globalYAdjusted = yStart * blockDim.y + threadIdx.y;
	int globalAdjusted = (globalIdY * 1920) + globalIdX;

	/////////////////
	// COMPUTATION //
	/////////////////

    float sum = 0.0;

    for (int i = 0; i < 32; i++)
        sum += inputs[globalAdjusted + i*1920];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int offset = 32/2; offset > 0; offset /= 2)
            sum += __shfl_down(sum, offset, 32);
        if (sum > 7000) {
            output[blockId] = 1;
        } else {
            output[blockId] = 0;
        }
    }
}

