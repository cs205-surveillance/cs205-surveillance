__global__ void superPixel(int *input, int *sum, int *output)
{
	
	//This kernel receives input values of 1's and 0's from gaussian running average/rPCA 
	//This kernel sums each pixels (i.e., the 1's) in each workgroup (superpixel)
	//This kernel outputs a single value that represents the sum of values in each superpixel

	//NOTES:
	//output is a 1 x [number of superpixels] array
	//sum is a 1 x [number of threads per block (blockDim.x)] array
	//input is a 1 x [number of pixels in image] array
	//blockDim.x may have to be set manually to a power of 2
	//need to set threshold per workgroup/block in driver to assess ouput values

	//initialize indexing values
	int idx = threadIdx.x + blockIdx.x * blockDim.x //global threadId
	int threadIdx = threadIdx.x                     //local threadId in a block
	int blockIdx = blockIdx.x                       //block index
	int blockDim = blockDim.x                       //threads per block

	//assign values from binary input value array to our local sum array
    sum[threadIdx] = input[idx]; 
    __syncthreads();

    //add up all 1's and 0's in local group using binary reduction
	for (size_t offset = blockDim/2; offset > 0 ; offset >>= 1) {
        if (threadIdx < offset) {    
            sum[threadIdx] += sum[threadIdx + offset];
        }
        __syncthreads();
    }
    //ouput final value
    if (threadIdx == 0) output[blockDim] = sum[0];
}

