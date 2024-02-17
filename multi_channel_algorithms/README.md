# multi channel algorithms 

# RTFs estimation

estimation of RTFs using GEVD.
the GEVD is computed using the covariance matrices of the signal and noise.
the signal and noise covariance matrices are computed using the signal and noise segments.
the algorithm assumes that the noise and signal segment is known.
input parameters:
- `X` - the signal segment
- `N` - the noise segment
- `M` - the number of microphones
- `L` - the number of samples in the signal and noise segments
- `P` - the number of sources
output:
- `w` - the RTFs

# MVDR beamforming

MVDR beamforming is a method to estimate the signal of interest using the noise covariance matrix and the RTFs.
the algorithm assume that the RTFs and the noise segment is known and given.
input:
- `Rxx` - the signal covariance matrix
- `Rnn` - the noise covariance matrix
- `Rxx` and `Rnn` are computed using the signal and noise segments.
- `w` - the RTFs
output:
- `y` - the estimated signal of interest

# LCMV beamforming

LCMV beamforming is a method to estimate the signal of interest using the noise covariance matrix and the RTFs.
the algorithm assume that the RTFs and the noise segment is known and given.



