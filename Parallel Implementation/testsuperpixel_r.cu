


#include <stdio.h>


__global__ void testsuperPixel(float *inputs, int *output)
{
	////////////////////
	// KERNEL OVERVIEW /
	////////////////////

	/*
	This kernel is used to determine if there is an anomaly in a superpixel.
	This kernel takes in a 1080x32 array (the output from minimum filter kernel) 
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

	/*
    int globalIdX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalId = (globalIdY * 32) + globalIdX;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int localId = (threadIdx.y * blockDim.x) + threadIdx.x;

    // Virtual 32 x 32
	int yStart = 32 * blockIdx.y;
	int globalYAdjusted = yStart * blockDim.y + threadIdx.y;
	int globalAdjusted = (globalIdY * 32) + globalIdX;
	*/



	///////////////////////////////////////////////////////////////////////////
	// AJ's SPACE:
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int globalIdY = 30 * blockIdx.y;
	int globalIdX = 32 * globalIdY + blockIdx.x * 32 + threadIdx.x;
	int globalId  = (globalIdY * 32) + globalIdX;

	float sum = 0.0;

	printf("%d\n",globalIdX);
	printf("%d\n",globalId);
	if (globalIdY < 4 && globalIdX < 32) {
			// Sum column of pixels below 
		for (int i =0; i <30; i++) {
			//printf("%f\n",sum);
			sum += inputs[globalId + i*32];
		}
	
	
		//works up until here here		
	__syncthreads();
	}
	// Sum all values in our block
    for (int offset = 16; offset > 0; offset /= 2) {
        //printf("%f\n",sum);
        sum += __shfl_down(sum, offset);
    	}
	//printf("%f\n",sum);	
	   
	if (globalId % 32 == 0) {
	    if (sum > 200) {
	        output[blockId] = 1;
	    } 
	    else {
	        output[blockId] = 0;
	    }   
	}
}
	///////////////////////////////////////////////////////////////////////////





	/////////////////
	// COMPUTATION //
	/////////////////
/*
    float sum = 0.0;

    if (globalAdjusted + 32 * 32 < 32 * 1080) {
        for (int i = 0; i < 32; i++)
            sum += inputs[globalAdjusted + i * 32]; // sum += inputs[(yStart * 32) + globalIdX + i*32];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int offset = 32/2; offset > 0; offset /= 2)
            sum += __shfl_down(sum, offset, 32); //may have to be "16"
        if (sum > 15*700) {
            output[blockId] = 1;
        } else {
            output[blockId] = 0;
        }
    }
}
*/