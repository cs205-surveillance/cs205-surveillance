# Notes for what works and what doesn't

#How do I copy a file from my local machine to the GPU?

in a separate terminal window type:
>> scp LOCAL_FILE_TO_COPY username@140.247.107.101:/DEST/FOLDER

#How do I copy a file from my the GPU to my local machine?

in a separate terminal window type:
>> scp username@140.247.107.101:/DEST/FOLDER/FILE DEST/FOLDER 

# What have we installed for everyone
Pip
Python
Scipy
Numpy

# What do we need to install still
pyCuda (we need CUDA installed)

