import numpy as np
from scipy import misc
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Import and compile CUDA kernels
rga_source = SourceModule(open('run_gaussian_average.cu').read())
run_gaussian_average = rga_source.get_function('run_gaussian_average')

superpixel_source = SourceModule(open('superpixel.cu').read())
run_super_pixel = superpixel_source.get_function('superPixel')

# filter_source = SourceModule(open('minimum_filter.cu').read())
# run_minimum_filter = filter_source.get_function('minimum_3x3')

# Loop over all images
for i in range(65, 90):

    # Number image according to file name
    image_number = str(i)
    while len(image_number) < 3:
        image_number = "0" + image_number

    # Load current image
    img = misc.imread('../../thouis/grabber{}.ppm'.format(image_number), flatten=True)
    img = img.astype(np.float32).reshape((1920*1080,))

    if i == 65:
        # As an initialization, set mu to initial image in stack
        mu_gpu = gpuarray.to_gpu(img)

        # As an initialization, set variance to 1 for each pixel
        sig2_gpu = gpuarray.zeros_like(mu_gpu) + 1

        # Initialize the output image
        rga_out_gpu = gpuarray.zeros_like(mu_gpu)

    # Copy image to device
    img_gpu = gpuarray.to_gpu(img)

    # Run Gaussian Average kernel
    run_gaussian_average(img_gpu, mu_gpu, sig2_gpu, rga_out_gpu, block=(15, 1, 1), grid=(1920*1080/15, 1))

    # Reshape RGA output from 1D to 2D
    rga_out_gpu = rga_out_gpu.reshape((1080, 1920))

    # Copy back (for testing)
    # sig2_result = sig2_gpu.get()
    # mu_result = mu_gpu.get().reshape((1080, 1920))
    #rga_result = rga_out_gpu.get().reshape((1080, 1920))

    #plt.imshow(rga_result)
    #plt.show()

    # Run 3x3 Minimum filter to remove speckle noise
    #run_minimum_filter()

    # Set parameters for super pixel kernel
    tol = np.array([.75])
    tol_gpu = gpuarray.to_gpu(tol)
    spxl_out = np.zeros((1920) * (1080), dtype=int)
    spxl_out_gpu = gpuarray.to_gpu(spxl_out)
    
    # Run super pixel kernel
    run_super_pixel(rga_out_gpu, tol_gpu, spxl_out_gpu, block=(30, 30, 1), grid=(1920/30, 1080/30))
    result = spxl_out_gpu.get().reshape((1080, 1920))

    # Show image, perhaps with pylab
    print result
    plt.imshow(result)
    plt.show()

