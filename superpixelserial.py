import numpy as np

def superpixel(incoords, num_super_pixels):
    pixels_per_super_pixel = len(incoords)/num_super_pixels
    superpixels=np.zeros(num_super_pixels)
    for i in range(num_super_pixels):
        for j in range(pixels_per_super_pixel):
            superpixels[i] += incoords[j+i*pixels_per_super_pixel]
    return superpixels/pixels_per_super_pixel


def detector(incoords,num_super_pixels,threshold):
	anomalies = np.zeros(num_super_pixels)
	for i in range(len(incoords)):
		if incoords[i] > threshold:
			anomalies[i] = 1
	return anomalies

inputs = np.zeros(100)
inputs[22:25]=1

threshold = .1
num_super_pixels = 10

output = superpixel(inputs,num_super_pixels)
anomalies = detector(output,num_super_pixels,threshold)
print anomalies

