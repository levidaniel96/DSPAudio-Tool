# Audio signal processing 

This repository contains several audio signal processing algorithms implemented in Python.

## create data set
This folder contains the code to create a dataset of audio files. The dataset is created using rir_generator and clean speech files. The clean speech files are convolved with the room impulse response to create the dataset. 

## T60 Estimation
This folder contains the code to estimate the reverberation time of a room impulse response. The reverberation time is estimated using the CNN model.

## multi channel algorithms 
This folder contains the code to implement several multi-channel algorithms. The algorithms implemented are:
- MVDR - Minimum variance distortionless response (Numpy and Pytorch)
- LCMV - Linearly constrained minimum variance (Numpy and Pytorch)
- RTF(Realtive Transfer Function) estimation using GEVD(Generalized Eigen Value Decomposition) 