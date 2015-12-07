import numpy as np
import matplotlib.pyplot as plt
from time import time
from coordinates import coordinates
from PIL import Image, ImageDraw
from scipy import misc
from serial_run_gaussian_average import rga
from serial_super_pixel import super_pixel
from serial_min_filter import min_filter
import os
home = os.getenv("HOME")

r = 30  # pixels, used to define superpixel dimension and outline
time_array_rga = []
time_array_min = []
time_array_sup = []
time_array = []

# Read in sequence of images, run them through RGA filter
for i in range(69, 300):
    image_number = str(i)
    while len(image_number) < 5:
        image_number = '0' + image_number

    I = misc.imread(home + '/../thouis/miscreants/{}.png'.format(image_number), flatten=True)
    I = I.astype(np.float32)

    t0 = time()

    # Initialize mu and variance based on initial frame.
    if i == 69: 
        mu = np.copy(I)
        sig2 = np.ones_like(I)

    # Run running gaussian average
    OUT, mu, sig2 = rga(I, 2.5, mu, sig2, cont=True) 
    t1 = time()
    
    # Save output from RGA
    # misc.imsave("cs205_images/mu_stack/mu_{}.jpeg".format(image_number), mu)
    # misc.imsave("cs205_images/sig2_stack/sig2_{}.jpeg".format(image_number), sig2)
    # misc.imsave("cs205_images/cont_output/cont_{}.jpeg".format(image_number), OUT)

    # Apply 3x3 minimum filter
    filt_out = min_filter(OUT, iterations=1)
    t2 = time()

    # Save output from 3x3 minimum filter
    # misc.imsave("cs205_images/filtered_output/filt_{}.jpeg".format(image_number), filt_out)

    # Aggregate into superpixels and flag anomalous behavior
    superpixel_output = super_pixel(filt_out, 12.5 * 700, r, r)
    t3 = time()
    output = coordinates(superpixel_output)

    # Track how much time each module took.
    time_array.append(t3 - t0)
    time_array_rga.append(t1 - t0)
    time_array_min.append(t2 - t1)
    time_array_sup.append(t3 - t2)

    #############################
    # OVERLAY ANOMALOUS REGIONS #
    #############################

    im = Image.open(home + '/../thouis/miscreants/{}.png'.format(image_number))
    draw = ImageDraw.Draw(im)
    numAnom = len(output)
    
    if numAnom > 0:
        for pt in output:
            # Draw rectangles
            draw.line((pt[1], pt[0], pt[1], pt[0] + r), fill=(255, 120, 0), width=4)
            draw.line((pt[1], pt[0] + r, pt[1] + r, pt[0] + r), fill=(255, 120, 0), width=4)
            draw.line((pt[1], pt[0], pt[1] + r, pt[0]), fill=(255, 120, 0), width=4)
            draw.line((pt[1] + r, pt[0], pt[1] + r, pt[0] + r), fill=(255, 120, 0), width=4)

    del draw
    im.save('../cs205_images/serial_output/tracker_{}.jpeg'.format(image_number))

print "Per frame processing time: ", np.mean(time_array)
print "Per frame rga time: ", np.mean(time_array_rga)
print "Per frame min_filt time: ", np.mean(time_array_min)
print "Per frame superpixel time: ", np.mean(time_array_sup)

