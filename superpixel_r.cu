__global__ void superPixel(float *inputs, int *output) {
	
	////////////////////
	// KERNEL OVERVIEW /
	////////////////////

	/*
	This kernel is used to determine if there is an anomaly in a superpixel.
	This kernel takes in a 1080x1920 array (the output from minimum filter kernel) 
	and calculates the sum for each 32x30 superpixel. Once the summation for every block 
	is complete, the kernel determines if the sum is above our tolerance. If so, the kernel 
	returns a 1 for that superpixel. Else, it returns a 0. The final output is a 1D array,
	where each value in that array corresponds to one of the superpixels.
	*/ 

	//Define parameters and thread variables
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int globalIdY = 30 * blockIdx.y;
	int globalIdX = blockIdx.x * 32 + threadIdx.x;
	int globalId  = (globalIdY * 1920) + globalIdX;
	float sum = 0.0;

	// Loop down 30 rows and sum
	for (int i = 0; i < 30; i++) {
		sum += inputs[globalId + i * 1920];
	}
	__syncthreads();

	// Sum across block using warp
	for (int offset = 16; offset > 0; offset /= 2) {
		sum += __shfl_down(sum, offset);
	}

	// Flag if above threshold
	if (threadIdx.x == 0) {
		if (sum > 15 * 700) {
			output[blockId] = 1;
		} else {
			output[blockId] = 0;
		}
	}
}
