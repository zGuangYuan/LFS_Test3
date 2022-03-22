Overview
Deep Learning-Based Heterogeneous Model  for ALN metastases prediction.

1¡¢System requirements:

Hardware Requirements
At least NVIDIA GTX 2080Ti

OS Requirements
This package is supported for Linux. The package has been tested on the following systems:
Linux: Ubuntu 16.04

Software Prerequisites
Python 3.6
Numpy 1.16.3
Scipy 1.2.1
CUDA 10.1
tensorflow-gpu 1.13.1
Pillow 4.0.0
opencv-python 3.4.0.12
Scikit-learn 0.20

2¡¢Installation guide:
It is recommended to install the environment in the Ubuntu 16.04 system.
First install Anconda3.
Then install CUDA 10.x and cudnn.
Finall intall these dependent python software library.
The installation is estimated to take 1 hour, depending on the network environment.

3¡¢Demo:

train model
python ./utils/train.py 

test model
python ./utils/test.py 


Note:
B-mode: B-mode US
SWE: shear wave elastography








