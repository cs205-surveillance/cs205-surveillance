import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def coordinates(inputs):

    # Reshape input array to 2D
    inputs = inputs.reshape(1080/30,1920/30)
    
    # Determine corresponding top-left corner pixel location for superpixels with 1's
    output = []
    for i in range(1080/30):
        for j in range(1920/30):
            if inputs[i,j]==1:
                point = [i*30,j*30]
                output.append(point)

    return output








