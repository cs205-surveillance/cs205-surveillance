#define N 512
int main(void) {
	int *a, *b, *c;   // host copies of a, b, c
	int *d_a, *d_b, *d_c; // Device copies of a, b, c
	int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	// Alloc space for host copies of a, b, c
	a = (int *)malloc(size); random_ints(a,N);
	b = (int *)malloc(size); random_ints(b,N);
	c = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N threads
	add<<<1,N>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	fprintf(c, "%s\n", );
	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}