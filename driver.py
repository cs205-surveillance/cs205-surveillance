import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


rga_source = SourceModule(open('run_gaussian_average.cu').read())
run_gaussian_average = rga_source.get_function('run_gaussian_average')

superpixel_source = SourceModule(open('superpixel.cu').read())
run_super_pixel = superpixel_source.get_function('superPixel')

filter_source = SourceModule(open('minimum_filter.cu').read())
run_minimum_filter = filter_source.get_function('minimum_3x3')


for i in range(65, 90):

    # Number image according to file name
    image_number = str(i)
    while len(image_number) < 3:
        image_number = "0" + image_number

    # Grab one image
    I = misc.imread('../thouis/grabber{}.ppm'.format(image_number), flatten=True)
    I = I.astype(np.float32).reshape((1920*1080,))

    if i == 65:
        # As an initialization, set mu to initial image in stack
        mu_gpu = gpuarray.to_gpu(I)

        # As an initialization, set variance to 1 for each pixel
        sig2_gpu = gpuarray.zeros_like(mu_gpu) + 1

        # Initialize the OUTPUT image
        OUT_gpu = gpuarray.zeros_like(mu_gpu)

    # Copy to device
    I_gpu = gpuarray.to_gpu(I)

    # Do algorithm
    run_gaussian_average(I_gpu, mu_gpu, sig2_gpu, OUT_gpu, block=(15, 1, 1), grid=(1920*1080/15, 1))

    # Copy back
    sig2_result = sig2_gpu.get()
    mu_result = mu_gpu.get().reshape((1080, 1920))
    rga_result = OUT_gpu.get().reshape((1080, 1920))

    plt.imshow(rga_result)
    plt.show()

    # Filter result
    run_minimum_filter()

    # Superpixel for object locations
    # inputs = gpuarray.to_gpu(OUT)
    # tol = np.array([.25])
    # TOL = gpuarray.to_gpu(tol)
    # out = np.zeros((1920/15)*(1080/15),dtype=int)
    # OUT2 = gpuarray.to_gpu(out)
    # run_super_pixel(OUT_gpu,TOL,OUT2, block=(15,15,1), grid=(1920/15,1080/15))
    # result = OUT2.get().reshape((1080/15,1920/15))

    # Show image, perhaps with pylab
    # print result
    # plt.imshow(result)
    # plt.show()

