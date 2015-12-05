import numpy as np
import matplotlib.pyplot as plt

from scipy import misc

def rga(I, thres, mu, sig2, rho=0.05, cont=False):
	# global mu, sig2
	
	# Loop through, pixel by pixel (or row by row), might be able to do whole matrix elementwise
	temp = abs(mu-I)/sig2 
	
	# Find which pixels are outside the threshold of the mean	
	anom_mask = temp > thres

	# For values that are within threshold, update mean and variance
	mu[~anom_mask] = rho * I[~anom_mask] + (1-rho) * mu[~anom_mask]
	d = abs(mu[~anom_mask]-I[~anom_mask])
	sig2[~anom_mask] = d * d * rho + (1-rho)*sig2[~anom_mask]
	
	if cont: # Check if you want continuous values saved rather than binary values.
		OUT = np.copy(temp)
	else: # Create binary mask on image based on any anomaly detected
		temp[~anom_mask] = 0
		temp[anom_mask] = 1
		OUT = np.copy(temp)

	return OUT, mu, sig2