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
    through the superpixels themselves within the input array
    
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
    imgWidth = incoords.shape[1]
    imgHeight = incoords.shape[0]

    #single superpixel width and height in terms of pixels
    blockDimC = width   
    blockDimR = height 

    #image dimensions in terms of superpixels
    gridDimC = imgWidth/blockDimC
    gridDimR = imgHeight/blockDimR

    #number of superpixels in image
    N = gridDimC*gridDimR
    superPixels = np.zeros(N)
    
    #loop and calculate sum of 1's in each superpixel
    n=0
    for r in range(gridDimR):
        for c in range(gridDimC):
            for i in range(blockDimR):
                for j in range(blockDimC):
                    superPixels[n] += incoords[i+blockDimR*r,j+blockDimC*c]
            n+=1
    
    #calculate percent of ones in each superpixel
    superPixels /= (blockDimR*blockDimC)        
    
    #assign 1 if percent > TOL, else assign 0
    outputA = np.zeros(N)
    for i in range(len(superPixels)):
        if superPixels[i] > TOL:
            outputA[i] = 1
    
    outputA = outputA.reshape(gridDimR,gridDimC)
    
    #take our grid of superpixels and output corresponding top-left corner pixel location
    outputB = []
    for i in range(gridDimR):
        for j in range(gridDimC):
            if outputA[i,j]==1:
                point = [i*blockDimR,j*blockDimC]
                outputB.append(point)

    return outputB

"""
########
# TEST #
########

#loop through set of images
for i in range(65,89):
    image_number = str(i)
    while len(image_number) < 3:
        image_number = '0' + image_number

    #load image
    I = misc.imread('cs205_images/Output/out{}.png'.format(image_number), flatten=True)
    I = I.astype(np.float32)

    #run superPixel fct
    output = superPixel(I,.8,30,30)

    #plot resulting output of superpixels
    plt.imshow(output)
    plt.show()
"""







