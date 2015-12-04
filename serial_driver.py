import numpy as np
import matplotlib.pyplot as plt
from time import time
from coordinates import coordinates
from PIL import Image, ImageDraw
from scipy import misc
from serial_run_gaussian_average import rga
from superpixelserial import superPixel
from serial_min_filter import min_filter

print "Made it through import statements"

r = 30 # pixels, used to define superpixel dimension and outline
time_array = []

# Read in sequence of images, run them through RGA filter
for i in range(69,300):
	image_number = str(i)
	while len(image_number) < 3:
		image_number = '0' + image_number

	I = misc.imread('cs205_images/grabber{}.ppm'.format(image_number), flatten=True)
	I = I.astype(np.float32)

	t0 = time()
	if i == 69:
		mu = np.copy(I)
		sig2 = np.ones_like(I)

	OUT, mu, sig2 = rga(I, 2.5, mu, sig2, cont=True)
	
	# Save output from RGA
	misc.imsave("cs205_images/mu_stack/mu_{}.jpeg".format(image_number), mu)
	misc.imsave("cs205_images/sig2_stack/sig2_{}.jpeg".format(image_number), sig2)
	misc.imsave("cs205_images/cont_output/cont_{}.jpeg".format(image_number), OUT)

	filt_out = min_filter(OUT, iterations=1)

	# Save output from 3x3 minimum filter
	misc.imsave("cs205_images/filtered_output/filt_{}.jpeg".format(image_number), filt_out)

	superpixel_output = superPixel(filt_out, 12.5*700,r,r)
	output = coordinates(superpixel_output)


	misc.imsave("cs205_images/superpixel_stack/sp_{}.jpeg".format(image_number), superpixel_output.reshape(1080/30.,1920/30.))

	t1 = time()
	time_array.append(t1-t0)

	im = Image.open('cs205_images/grabber{}.ppm'.format(image_number))
	draw = ImageDraw.Draw(im)
	numAnom = len(output)
	
	if numAnom > 0:
		for pt in output:
			#Draw rectangles
			draw.line((pt[1],pt[0],pt[1],pt[0]+r), fill=(255,120,0), width=4)
			draw.line((pt[1], pt[0]+r, pt[1]+r, pt[0]+r), fill=(255,120,0), width=4)
			draw.line((pt[1], pt[0], pt[1]+r, pt[0]), fill=(255,120,0), width=4)
			draw.line((pt[1]+r, pt[0], pt[1]+r, pt[0]+r), fill=(255,120,0), width=4)

	del draw
	im.save('cs205_images/serial_output/tracker_{}.png'.format(image_number))

print "Per frame processing time: ", np.mean(time_array)