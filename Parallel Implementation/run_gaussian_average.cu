__global__ void run_gaussian_average(float *I, float *mu, float *sig2, float *OUT) {

    /*
	 I = input image, intensities
	 mu = running average intensity for each pixel, initially set to 1st image
	 sig2 = running average variance for each pixel, initially set to 1
	 thres = threshold for comparison with mean value
	 OUT = output image with filtered values for each pixel [1 if foreground, 0 if background]
	 */

	// rho is a temporal parameter, used when updating the mean and variance
	float rho = 0.01;  // Increased from 0.01 to more quickly integrate slight variances in background
	float threshold = 2.5; 
	int SIZE = 1920 * 1080;

	// Get current idx
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < SIZE) {

		// Compare z-value with threshold. If below, update pixel mean and variance 
		if ((abs(I[idx] - mu[idx]) / sig2[idx]) - threshold < 0) {

			float d = abs(I[idx] - mu[idx]); // Deviation from mean
			mu[idx] = rho * I[idx] + (1 - rho) * mu[idx]; // Update pixel
			sig2[idx] = d*d * rho + (1 - rho) * sig2[idx]; // Update variance

		} 

		OUT[idx] = (abs(I[idx] - mu[idx]) / sig2[idx]); // Continuous output
	}
}