import numpy as np
from scipy import misc
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from coordinates import coordinates
from PIL import Image, ImageDraw
from time import time

# Import and compile CUDA kernels
rga_source = SourceModule(open('run_gaussian_average.cu').read())
run_gaussian_average = rga_source.get_function('run_gaussian_average')

superpixel_source = SourceModule(open('superpixel.cu').read())
run_super_pixel = superpixel_source.get_function('superPixel')

filter_source = SourceModule(open('minimum_filter.cu').read())
run_minimum_filter = filter_source.get_function('minimum_3x3')

t0 = time()
# Loop over all images
for i in range(71, 90):

    # Number image according to file name
    image_number = str(i)
    while len(image_number) < 3:
        image_number = "0" + image_number

    # Load current image
    img = misc.imread('../../thouis/grabber{}.ppm'.format(image_number), flatten=True)
    img = img.astype(np.float32).reshape((1, 1920 * 1080))

    # Initialization
    if i == 71:
        # Set mu to initial image in stack
        mu_gpu = gpuarray.to_gpu(img)

        # Set variance to 1 for each pixel
        sig2_gpu = gpuarray.zeros_like(mu_gpu) + 1

        # Initialize the output image
        rga_out_gpu = gpuarray.zeros_like(mu_gpu)

    # Copy image to device
    img_gpu = gpuarray.to_gpu(img)

    # Run Gaussian Average kernel
    run_gaussian_average(img_gpu, mu_gpu, sig2_gpu, rga_out_gpu, block=(15, 1, 1), grid=(1920 * 1080 / 15, 1))

    # Reshape RGA output from 1D to 2D
    rga_out_gpu = rga_out_gpu.reshape(1080, 1920)
    
    # Run 3x3 Minimum filter to remove speckle noise
    denoised_gpu = gpuarray.empty_like(rga_out_gpu)
    run_minimum_filter(rga_out_gpu, denoised_gpu, block=(3, 3, 1), grid=(1920, 1080))

    # Show image
    result = denoised_gpu.get()
    print(result)
    plt.imshow(result)
    plt.show()

    # Set parameters for super pixel kernel
    spxl_out = np.zeros((1920 / 30) * (1080 / 30), dtype=int)
    spxl_out_gpu = gpuarray.to_gpu(spxl_out)
    
    # Run super pixel kernel
    run_super_pixel(denoised_gpu, spxl_out_gpu, block=(30, 30, 1), grid=(1920 / 30, 1080 / 30))
    result = spxl_out_gpu.get()


    # Show image
    # print result.reshape((1080 / 30, 1920 / 30))
    # plt.imshow(result.reshape((1080 / 30, 1920 / 30)))
    # plt.show()
    
    output = coordinates(result)
    im = Image.open('../../thouis/grabber{}.ppm'.format(image_number))
    draw = ImageDraw.Draw(im)
    numAnom = len(output)
    r = 30
    if numAnom > 0:
        for pt in output:
            # Draw rectangles
            draw.line((pt[1],pt[0],pt[1],pt[0]+r), fill=(255,120,0), width=4)
            draw.line((pt[1], pt[0]+r, pt[1]+r, pt[0]+r), fill=(255,120,0), width=4)
            draw.line((pt[1], pt[0], pt[1]+r, pt[0]), fill=(255,120,0), width=4)
            draw.line((pt[1]+r, pt[0], pt[1]+r, pt[0]+r), fill=(255,120,0), width=4)

    del draw
    im.show()
# tend = time()
# print "Per frame processing time: ", (tend-t0)/(90-69)
