import numpy as np
import matplotlib.pyplot as plt

from scipy import misc

def rga(I, thres, rho=0.01):
	global mu, sig2
	
	# Loop through, pixel by pixel (or row by row), might be able to do whole matrix elementwise
	OUT = abs(mu-I)/sig2 # Trying out vectorized option first..., returning spectrum of changed values
	
	# Find which pixels are outside the threshold of the mean	
	anom_mask = OUT > thres

	# For values that are within threshold, update mean and variance
	mu[~anom_mask] = rho * I[~anom_mask] + (1-rho) * mu[~anom_mask]
	d = abs(mu[~anom_mask]-I[~anom_mask])
	sig2[~anom_mask] = d * d * rho + (1-rho)*sig2[~anom_mask]
	
	return OUT

global mu, sig2
# Read in sequence of images, run them through RGA filter
for i in range(2,20):
	image_number = str(i)
	while len(image_number) < 3:
		image_number = '0' + image_number

	I = misc.imread('cs205_images/grabber{}.ppm'.format(image_number), flatten=True)
	I = I.astype(np.float32)

	if i == 2:
		mu = np.copy(I)
		sig2 = np.ones_like(I)


	OUT = rga(I, 2.5)
	print "Number of non-zero entries: ", np.count_nonzero(OUT)
	plt.spy(OUT)
	plt.show()