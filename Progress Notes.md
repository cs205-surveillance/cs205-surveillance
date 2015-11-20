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
4) Classify Pixels (1/0)
5) Update Mean (for background only)
6) Potential Filter - denoise
7) Superpixel
8) Identify Regions above threshold
9) Drawing overlay 
10) Realtime check

# Next Steps
1) Learn how to Read PPM (Taylor)
2) Greyscale (Taylor)
3) Driver (Reinier)
4) Averaging Kernal (Taylor)
5) Superpixel/segment (Andrew)

Meet Friday after seminar to review.

