import numpy as np
from scipy import misc
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

test_superpixel_source = SourceModule(open('testsuperpixel.cu').read())
run_super_pixel = test_superpixel_source.get_function('testsuperPixel')

# filter_source = SourceModule(open('minimum_filter.cu').read())
# run_minimum_filter = filter_source.get_function('minimum_3x3')

test_array = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2]])
test_array_gpu = gpuarray.to_gpu(test_array)
print test_array

spxl_out = np.zeros(4, dtype=int)
spxl_out_gpu = gpuarray.to_gpu(spxl_out)

# Run super pixel kernel
run_super_pixel(test_array_gpu, spxl_out_gpu, block=(2, 2, 1), grid=(2, 2))
result = spxl_out_gpu.get().reshape((2,2))

# Show image, perhaps with pylab
print result


