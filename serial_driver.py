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

# global mu, sig2
# Read in sequence of images, run them through RGA filter
t0 = time()
for i in range(69,300):
	image_number = str(i)
	while len(image_number) < 3:
		image_number = '0' + image_number

	I = misc.imread('cs205_images/grabber{}.ppm'.format(image_number), flatten=True)
	I = I.astype(np.float32)

	if i == 69:
		mu = np.copy(I)
		sig2 = np.ones_like(I)


	OUT, mu, sig2 = rga(I, 2.5, mu, sig2, cont=True)
	# print "Number of non-zero entries: ", np.count_nonzero(OUT)
	# plt.imshow(OUT)
	# plt.show()

	filt_out = min_filter(OUT, iterations=1)

	# misc.imsave('cs205_images/cont_output/cont_out{}.png'.format(image_number),OUT)
	superpixel_output = superPixel(filt_out, 10*700,r,r)
	output = coordinates(superpixel_output)
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
	im.save('cs205_images/serial_output/tracker{}.png'.format(image_number))
tend = time()
print "Per frame processing time: ", (tend-t0)/(90-65)