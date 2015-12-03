import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def superPixel(incoords,TOL,width=15,height=15):
    
    """
    ###########
    # Summary #
    ###########
    This is the serial implementation of the superpixel calculations.
    The fct takes in a 2D binary array and computes parameters re: the img.
    4 for loops are required to loop in 2 dimensions through the
    pixels within an individual superpixel and loop in 2 dimensions
    through the superpixel themselves within the input array
    
    #########
    # Input #
    #########

    incoords = 1080 x 1920 array of 1's and 0's (e.g., the output from run_gaussian_avg)

    TOL = tolerance level for anomaly detection. if percent of 1's
    in a superpixel > TOL, then return 1 indicating an anomaly.
    
    width = desired length of superpixel, default set to 15
    
    height = desired width of superpixel, default set 15

    ##########
    # Output #
    ##########
    
    Returns a 2D array of 1's and 0's indicating anomaly for each superpixel. 

    """

    #image dimensions in terms of pixels
    img_width = incoords.shape[1]
    img_height = incoords.shape[0]

    #single superpixel width and height in terms of pixels
    block_dim_c = width   
    block_dim_r = height 

    #image dimensions in terms of super_pixel
    grid_dim_c = img_width/block_dim_c
    grid_dim_r = img_height/block_dim_r

    #number of super_pixel in image
    N = grid_dim_c*grid_dim_r
    super_pixel = np.zeros(N)
    
    #loop and calculate sum of 1's in each superpixel
    n=0
    for r in range(grid_dim_r):
        for c in range(grid_dim_c):
            for i in range(block_dim_r):
                for j in range(block_dim_c):
                    super_pixel[n] += incoords[i+block_dim_r*r,j+block_dim_c*c]
            n+=1
    
    #calculate percent of ones in each superpixel
    # super_pixel /= (block_dim_r*block_dim_c)        
    
    #assign 1 if percent > TOL, else assign 0
    output_intermediate = np.zeros(N)
    for i in range(len(super_pixel)):
        if super_pixel[i] > TOL:
            output_intermediate[i] = 1
    
    output_intermediate = output_intermediate.reshape(grid_dim_r,grid_dim_c)
    
    #take our grid of super_pixel and output corresponding top-left corner pixel location
    output_final = []
    for i in range(grid_dim_r):
        for j in range(grid_dim_c):
            if output_intermediate[i,j]==1:
                point = [i*block_dim_r,j*block_dim_c]
                output_final.append(point)

    return output_final







