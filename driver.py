import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from scipy import misc

source = SourceModule(open('run_gaussian_average.cu').read())
run_gaussian_average = source.get_function('run_gaussian_average')

# Grab one image
# grabber001 => np.array
# I = np.random.randn(1920, 1080)
I = misc.imread('../thouis/grabber000.ppm',flatten=True)

# Copy to device
I = I.astype(np.float32)
I_gpu = cuda.mem_alloc(I.nbytes)
cuda.memcpy_htod(I_gpu, I)

# mu = np.zeros_like(I)
mu = np.copy(I)               # As an initialization, set mu to initial image in stack
mu_gpu = cuda.mem_alloc(mu.nbytes)
cuda.memcpy_htod(mu_gpu, mu)

sig2 = np.ones_like(I)        # As an initialization, set variance to 1 for each pixel
sig2_gpu = cuda.mem_alloc(sig2.nbytes)
cuda.memcpy_htod(sig2_gpu, sig2)

OUT = np.zeros_like(I)
OUT_gpu = cuda.mem_alloc(OUT.nbytes)
cuda.memcpy_htod(OUT_gpu, OUT)

# Do algorithm

run_gaussian_average(cuda.In(I), cuda.InOut(mu), cuda.InOut(sig2), cuda.Out(OUT))

# Copy back
cuda.memcpy_dtoh(mu_gpu, mu)
cuda.memcpy_dtoh(sig2_gpu, sig2)
cuda.memcpy_dtoh(OUT_gpu, OUT)

# Post process


# Show image, perhaps with pylab
print OUT

