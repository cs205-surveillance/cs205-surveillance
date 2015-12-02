# Notes for what works and what doesn't

#How do I copy a file from my local machine to the GPU?
in a separate terminal window type:
>> scp LOCAL_FILE_TO_COPY username@140.247.107.101:/DEST/FOLDER

#How do I copy a file from my the GPU to my local machine?
in a separate terminal window type:
>> scp username@140.247.107.101:/DEST/FOLDER/FILE  LOCAL_DEST/FOLDER 

# What have we installed for everyone
Pip
Python
Scipy
Numpy
pyCuda (Reinier installed)

# What do we need to install still


# Methodology
1) Read Image
2) Filter Image
3) Compare with Mean
4) Classify Pixels (1/0) [ray notes]: do some soft thing, not necessarily 1/0's. you dont just do a distance you do distance relative to the array for gaussian. so instead of 0 and 1 we instead do what ever that distance is in terms of standard deviations and can be arbitrarily high and may want to clamp at 3 stds. its a multidimensional z score. for pca you don't get the same thing. but you need to take into account the shape. do continuous output where 0 is exactly mean and anything away is a number. . 
5) Update Mean (for background only)
6) Potential Filter - denoise
7) Superpixel - slic is more expensive and more interesting to write. and can use teh grid for detection. slic would give better boundary. 
8) Identify Regions above threshold
9) Drawing overlay 
10) Realtime check

# Notes from Ray meeting 11/20
1) added cuda path for andrew (and others). nvcc should work.
2) export is temporary. 
3) may need to add contrast adaption with images
4) timeset = a timeout if it can read anything. but otherwise, it just grabs. captures 20 frames per second. 
5) ssh with capital Y to display back. do ssh Y to see visual
6) over break do the rpca or running average. 
7) choose either slic or grid method. depends on how we get to that step

# Notes from Ray Meeting 12/2
what does the variance look like
go with continuous not binary
curious: why does sign look dark
cut out small region of sign and plot how it progresses with time
small window looking at avg. intensity and avg. variance on 3x3 pixel.
continuous output easier to use.
minimum filter on continous 3x3 filter to get rid of spec noises
avg threshold per box. 
minimum filter is take 3x3 filter like median filter nad take lowest value not median value. 
use numpy as a view using a slice and set four borders to a color. 
check pycuda make sure yuo aren't copying from GPU memory before copying
want to minimize readbacks.