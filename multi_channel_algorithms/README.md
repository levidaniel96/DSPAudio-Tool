# multi channel algorithms 

# RTFs estimation

estimation of RTFs using GEVD.
the GEVD is computed using the covariance matrices of the signal and noise.
the signal and noise covariance matrices are computed using the signal and noise segments.
the algorithm assumes that the noise and signal segment is known.
input parameters:
- `input` - the noisy signal 
- `start_time_spk` - the start time of the signal segment
- `end_time_spk` - the end time of the signal segment
- `start_time_noise` - the start time of the noise segment
- `end_time_noise` - the end time of the noise segment
- `M` - the number of microphones
- STFT parameters such as the window size, overlap, and the number of FFT points.
- `Nl` - length of the RTF cut from the left
- `Nr` - length of the RTF cut from the right 
- `ref_mic` - the reference microphone
output:
- `RTFs` - the RTFs in time domain

# MVDR beamforming - numpy implementation

MVDR beamforming is a method to estimate the signal of interest using the noise covariance matrix and the RTFs.
the algorithm assume that the noise and signal segment is known.
input:
- `input` - the noisy signal 
- `start_time_spk` - the start time of the signal segment
- `end_time_spk` - the end time of the signal segment
- `start_time_noise` - the start time of the noise segment
- `end_time_noise` - the end time of the noise segment
- `M` - the number of microphones
- STFT parameters such as the window size, overlap, and the number of FFT points.
- `Nl` - length of the RTF cut from the left
- `Nr` - length of the RTF cut from the right
- `ref_mic` - the reference microphone

output:
- `y` - the estimated signal of interest


# MVDR beamforming - pytorch implementation

MVDR beamforming is a method to estimate the signal of interest using the noise covariance matrix and the RTFs.
the algorithm assume that the noise and signal segment is known and the RTFs are given.
input:
- `input` - the noisy signal (batch, time, M)
- `RTFs` - the RTFs (batch, M-1, Nl+Nr+1) the reference not included
- `M` -  number of microphones
- `start_time_noise` - the start time of the noise segment 
- `end_time_noise` - the end time of the noise segment
- STFT parameters such as the window size, overlap, and the number of FFT points.

output:
- `y` - the estimated signal of interest (batch, time, 1)



# LCMV beamforming

LCMV beamforming is a method to estimate the signal of interest using the noise covariance matrix and the RTFs.
the algorithm assume that the RTFs and the noise segment is known and given.



