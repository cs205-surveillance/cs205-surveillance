import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def super_pixel(incoords,TOL,width=30,height=30):
    
    """
    #####################
    # Function Overview #
    #####################

    This function is used to determine if there is an anomaly in a superpixel.
    This function takes in a 1080x1920 array (the output from minimum filter function) 
    and calculates the sum for each 30x30 pixel block (i.e., superpixel). Once the 
    summation for every block is complete, the function determines if the sum is above our
    tolerance. If so, the function returns a 1 for that superpixel. Else, it returns a 0. The 
    final output is a 1D array that includes the 1's and 0's for all superpixels.
    
    """

    #########################
    # Initialize Parameters #
    #########################

    # Image dimensions in terms of pixels
    img_width = incoords.shape[1]
    img_height = incoords.shape[0]

    # Single superpixel width and height in terms of pixels
    block_dim_c = width   
    block_dim_r = height 

    # Image dimensions in terms of super_pixel
    grid_dim_c = img_width/block_dim_c
    grid_dim_r = img_height/block_dim_r

    # Number of super_pixel in image
    N = grid_dim_c*grid_dim_r
    super_pixel = np.zeros(N)
    
    ###############
    # Computation #
    ###############

    # Calculate sum of values in each superpixel
    n=0
    for r in range(grid_dim_r):
        for c in range(grid_dim_c):
            for i in range(block_dim_r):
                for j in range(block_dim_c):
                    super_pixel[n] += incoords[i+block_dim_r*r,j+block_dim_c*c]
            n+=1
         
    # Assign 1 if sum > tolerance
    output = np.zeros(N)
    for i in range(len(super_pixel)):
        if super_pixel[i] > TOL:
            output[i] = 1
    
    return output






