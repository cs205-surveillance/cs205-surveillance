import numpy as np

def superPixel(incoords,length,width,TOL):
    
    #image dimensions in terms of pixels
    imgWidth = incoords.shape[1]
    imgHeight = incoords.shape[0]

    #single superpixel width and height in terms of pixels
    blockDimC = width
    blockDimR = length

    #image dimensions in terms of superpixels
    gridDimC = imgWidth/blockDimC
    gridDimR = imgHeight/blockDimR

    #number of superpixels in image
    N = gridDimC*gridDimR
    superPixels = np.zeros(N)
    
    #loop and calculate sum of 1's in each superPixel
    n=0
    for r in range(gridDimR):
        for c in range(gridDimC):
            for i in range(blockDimR):
                for j in range(blockDimC):
                    superPixels[n] += incoords[i+blockDimR*r,j+blockDimC*c]
            n+=1
    #calculate percent of ones in superpixel
    superPixels /= (blockDimR*blockDimC)
    
    #assign 1 if percent > TOL, else assign 0
    output = np.zeros(N)
    for i in range(len(superPixels)):
        if superPixels[i] > TOL:
            output[i] = 1
    return output
