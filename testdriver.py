import numpy as np
from scipy import misc
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

test_superpixel_source = SourceModule(open('testsuperpixel_r.cu').read())
run_super_pixel = test_superpixel_source.get_function('testsuperPixel')

test_array = np.array([[10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,8010,20,30,40,50,60,70,80,10,20,30,40,50,60,70,8010,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80],[10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,8010,20,30,40,50,60,70,80,10,20,30,40,50,60,70,8010,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80],[10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80],[10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80,10,20,30,40,50,60,70,80]]).reshape(4,32)
test_array_gpu = gpuarray.to_gpu(test_array)
print test_array

spxl_out = np.zeros(1, dtype=int)
spxl_out_gpu = gpuarray.to_gpu(spxl_out)

# Run super pixel kernel
# for grid put the number of blocks across and then blocks down.
run_super_pixel(test_array_gpu, spxl_out_gpu, block=(32, 1, 1), grid=(1,1))
result = spxl_out_gpu.get()#.reshape((8,16))

# Show image, perhaps with pylab
print result


