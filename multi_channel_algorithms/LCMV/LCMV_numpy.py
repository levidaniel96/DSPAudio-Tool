import numpy as np
import scipy.signal as ss



def LCMV_RTFs(y,RTFs,Qvv,params):
    """
    LCMV_RTFs: LCMV beamforming algorithm for multi-channel speech enhancement
    Inputs:
    y: noisy speech signal (n_samples, n_channels)
    RTFs: Room Transfer Functions (n_frequencies, n_channels, n_sources)
    Qvv: Noise covariance matrix (n_frequencies, n_channels, n_channels)
    params: parameters of the algorithm
    Outputs:
    first_channel: enhanced signal for the first channel
    second_channel: enhanced signal for the second channel
    """
    
    frame_count = 1 + (y.shape[0] - params.wlen ) // params.n_hop

    Y_STFT_matrix=np.zeros([params.NUP,frame_count,params.M],dtype=complex)

    for m in range(params.M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],params.fs, np.hamming(params.wlen) , nperseg=params.wlen, noverlap=params.overlap, nfft=params.NFFT,boundary=None,padded=False)[2] 

    output_y_stft = np.zeros([params.NUP,2,frame_count], dtype=complex)
    e=1e-6
    w=np.zeros([params.M,2,params.NUP],dtype=complex)
    for k in range(params.NUP):
        g = (RTFs[k,:,:].squeeze())
        b = (Qvv[k,:,:].squeeze())
        inv_b = b + e*np.linalg.norm(b)*np.eye(params.M)
        c = np.matmul(np.linalg.inv(inv_b),g)
        g_conj = np.conj(g).T
        inv_temp =  np.matmul(g_conj,c) + e*np.linalg.norm(np.matmul(g_conj,c))*np.eye(2)
        w[:,:,k] = (np.matmul(c , np.linalg.inv(inv_temp)))
        output_y_stft[k,:,:] = np.matmul(np.conj(w[:,:,k].squeeze()).T,(Y_STFT_matrix[k,:,:].squeeze()).T)
    first_channel=ss.istft(output_y_stft[:,0,:].squeeze(),params.fs,window=np.hamming(params.wlen), nperseg=params.wlen, noverlap=params.overlap, nfft=params.NFFT)[1]
    second_channel=ss.istft(output_y_stft[:,1,:].squeeze(),params.fs,window=np.hamming(params.wlen), nperseg=params.wlen, noverlap=params.overlap, nfft=params.NFFT)[1]
        
    return first_channel, second_channel
