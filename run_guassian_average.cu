__global__ void run_guassian_average(float *I, float *mu, float *sig2, float *OUT)
{
	// I = input image, intensities
	// mu = running average intensity for each pixel, initially set to 1st image
	// sig2 = running average variance for each pixel, initially set to 1
	// thres = threshold for comparison with mean value
	// OUT = output image with filtered values for each pixel [1 if foreground, 0 if background]

	// rho is a temporal parameter, used when updating the mean and variance
	float rho = 0.01;
	float threshold = 1.96;

	// DO I NEED TO DECLARE 'OUT' as __shared__ ??

	// Get current idx
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Compare abs(I[idx]-mu[idx])/sig[idx] < thres
	if(abs(I[idx]-mu[idx])/sig2[idx] < threshold) {
		// If True, mark OUT[idx] = 1
		OUT[idx] = 1;
	}
	else {
		// Else, mark OUT[idx] = 0, adjust mean and variance
		OUT[idx] = 0;
		float d = abs(I[idx]-mu[idx]);
		mu[idx] = rho * I[idx] + (1 - rho) * mu[idx];
		sig2[idx] = d*d * rho + (1 - rho) * sig2[idx];
	}

}