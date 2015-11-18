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

# What do we need to install still
pyCuda (we need CUDA installed) 

# Next steps
1) get image onto local machine
2) investigate image properties
3) on GPU run something ...
4) Look into file explorer link to SSH location

