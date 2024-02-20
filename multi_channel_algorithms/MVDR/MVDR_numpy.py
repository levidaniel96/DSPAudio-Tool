
import numpy as np
import scipy.signal as ss
from numpy import linalg as LA

def MVDR(y,RTFs,Qvv,params):
    
    frame_count = 1 + (y.shape[0] - params.wlen ) // params.n_hop
    epsilon=np.finfo(np.float16).eps
    Y_STFT_matrix=np.zeros([params.NUP,frame_count,params.M],dtype=complex)
    for m in range(params.M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],params.fs, np.hamming(params.wlen) , nperseg=params.wlen, noverlap=params.overlap, nfft=params.NFFT,boundary=None,padded=False)[2] 

    output_y_stft = np.zeros([params.NUP,frame_count], dtype=complex)
    w=np.zeros([params.M,params.NUP],dtype=complex)
    for f in range(params.NUP):
        inv_qvv = LA.inv(np.squeeze(Qvv[f, :, :]) + epsilon * LA.norm(Qvv[f, :, :]) * np.eye(params.M))      
        b = inv_qvv @ RTFs[f,:]    
        denom = np.squeeze(RTFs[f, :]).conj().T @ b
        w = b / denom
        output_y_stft[f,:]=w.conj() @ np.squeeze(Y_STFT_matrix[f,:,:]).T
        
    y_n = ss.istft(output_y_stft, params.fs, np.hamming(params.wlen), nperseg=params.wlen, noverlap=params.overlap, nfft=params.NFFT)[1]
        
    return y_n

