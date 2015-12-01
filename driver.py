import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy import misc
import time
import matplotlib.pyplot as plt

source = SourceModule(open('run_gaussian_average.cu').read())
run_gaussian_average = source.get_function('run_gaussian_average')

source1 = SourceModule(open('superpixel.cu').read())
run_super_pixel = source1.get_function('superPixel')

# mu = np.zeros_like(I)

for i in range(70,90):
	image_number = str(i)
	while len(image_number) < 3:
		image_number = "0"+image_number

	# Grab one image
	I = misc.imread('../../thouis/grabber{}.ppm'.format(image_number), flatten=True)
	I = I.astype(np.float32)

	if i == 70:
		mu = np.copy(I)               # As an initialization, set mu to initial image in stack
		mu_gpu = cuda.mem_alloc(mu.nbytes)
		cuda.memcpy_htod(mu_gpu, mu)

		sig2 = np.ones_like(I)        # As an initialization, set variance to 1 for each pixel
		sig2_gpu = cuda.mem_alloc(sig2.nbytes)
		cuda.memcpy_htod(sig2_gpu, sig2)

		OUT = np.zeros_like(I)
		OUT_gpu = cuda.mem_alloc(OUT.nbytes)
		cuda.memcpy_htod(OUT_gpu, OUT)

	# Copy to device
	I_gpu = cuda.mem_alloc(I.nbytes)
	cuda.memcpy_htod(I_gpu, I)

	# Do algorithm
	run_gaussian_average(I_gpu, mu_gpu, sig2_gpu, OUT_gpu, block=(15,15,1), grid=(1920/15,1080/15))

	# Copy back
	#cuda.memcpy_dtoh(mu,mu_gpu)
	#cuda.memcpy_dtoh(sig2,sig2_gpu)
	cuda.memcpy_dtoh(OUT,OUT_gpu)

	print "Number of non-zero entries: ", np.count_nonzero(OUT)
	plt.imshow(OUT)
	plt.show()

	# Post process
	#inputs = gpuarray.to_gpu(OUT)
	tol = np.array([.25])
	TOL = gpuarray.to_gpu(tol)
	out = np.zeros((1920/15)*(1080/15),dtype=int)
	OUT2 = gpuarray.to_gpu(out)
	run_super_pixel(OUT_gpu,TOL,OUT2, block=(15,15,1), grid=(1920/15,1080/15))
	result = OUT2.get().reshape((1080/15,1920/15))

	# Show image, perhaps with pylab
	print result
	plt.imshow(result)
	plt.show()

