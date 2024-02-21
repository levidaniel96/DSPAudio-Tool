# multi channel algorithms 

## RTFs estimation

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


## MVDR beamforming

MVDR (Minimum Variance Distortionless Response) beamforming is a method used to estimate the signal of interest using the noise covariance matrix and the RTFs.
The algorithm assumes that the noise and signal segments are known in the numpy version. 
and assume that the covariance matrix of the noise is known in the pytorch version.

### MVDR beamforming - numpy implementation

input:
- `input` - the noisy signal 
- `start_time_spk` - the start time of the signal segment
- `end_time_spk` - the end time of the signal segment
- `start_time_noise` - the start time of the noise segment
- `end_time_noise` - the end time of the noise segment
- `M` - the number of microphones
- `params` - a dictionary containing the STFT parameters such as the window size, overlap, and the number of FFT points.


output:
- `y` - the estimated signal of interest


### MVDR beamforming - pytorch implementation

input:
- `input` - the noisy signal (batch, time, M)
- `RTFs` - the RTFs (batch, M-1, Nl+Nr+1) the reference not included
- `Qvv` - the noise covariance matrix (batch, M, M)
- `start_time_noise` - the start time of the noise segment 
- `end_time_noise` - the end time of the noise segment
- `params` - a dictionary containing the STFT parameters such as the window size, overlap, and the number of FFT points.

output:
- `y` - the estimated signal of interest (batch, time, 1)



## LCMV beamforming

LCMV (Linearly Constrained Minimum Variance) beamforming is a method used to estimate the signal of interest using the noise covariance matrix and the RTFs.
The algorithm assumes that the RTFs and the noise correlation matrix are known and provided. 
This implementation supports 2 speakers but can be easily extended to more speakers.

### LCMV beamforming - numpy implementation

input:
- `input` - the noisy signal
- `RTFs` - the RTFs in frequency domain for the first and second speaker (M, K)
- `Qvv` - the noise covariance matrix (batch, M, M)
- `params` - a dictionary containing the STFT parameters such as the window size, overlap, and the number of FFT points.

output:
- `first_channel` - the estimated signal of interest for the first speaker
- `second_channel` - the estimated signal of interest for the second speaker

### LCMV beamforming - pytorch implementation

input:
- `input` - the noisy signal (batch, time, M)
- `h_first_spk` - the RTFs in time domain for the first speaker (batch, M, wlen)
- `h_second_spk` - the RTFs in time domain for the second speaker (batch, M, wlen)
- `Qvv` - the noise covariance matrix (batch, M, M)
- `params` - a dictionary containing the STFT parameters such as the window size, overlap, and the number of FFT points.

output:
- `first_channel` - the estimated signal of interest for the first speaker
- `second_channel` - the estimated signal of interest for the second speaker    



