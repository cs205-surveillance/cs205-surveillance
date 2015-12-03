import numpy as np
from scipy import misc
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

test_superpixel_source = SourceModule(open('testsuperpixel.cu').read())
run_super_pixel = test_superpixel_source.get_function('testsuperPixel')

test_array = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2]])
test_array_gpu = gpuarray.to_gpu(test_array)
print test_array

spxl_out = np.zeros(8, dtype=int)
spxl_out_gpu = gpuarray.to_gpu(spxl_out)

# Run super pixel kernel
run_super_pixel(test_array_gpu, spxl_out_gpu, block=(4, 4, 1))
result = spxl_out_gpu.get()#.reshape((8,16))

# Show image, perhaps with pylab
print result


