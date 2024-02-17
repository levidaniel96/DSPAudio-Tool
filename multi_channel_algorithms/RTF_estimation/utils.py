import numpy as np
import math

def creat_Qvv_Qzz(Y_STFT_matrix,params,noise_frm_st=0,noise_frm_fn=0,first_frm_st=0,first_frm_fn=0):
    '''
    This function creats the noise and speech covariance matrices
    Input: 
        Y_STFT_matrix: STFT of the signal
        params: parameters  
    Output:
        Rzz: noise covariance matrix
        Rvv: speech covariance matrix
    '''
    Rzz=np.zeros([int(eval(params.NUP)),params.M,params.M],dtype=complex)
    Rvv=np.zeros([int(eval(params.NUP)),params.M,params.M],dtype=complex)
    noise_frm_st = int(math.ceil(0*params.fs/params.n_hop))
    noise_frm_fn = int(math.floor(2*params.fs/params.n_hop-1))
    first_frm_st = int(math.ceil(2*params.fs/params.n_hop))
    first_frm_fn = int(math.floor(4*params.fs/params.n_hop)) 
    for k in range(int(eval(params.NUP))):
        Rvv[k,:,:]=Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].T@Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].conj()/len(Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:])
        Rzz[k,:,:]=Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].T@Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].conj()/len(Y_STFT_matrix[k,first_frm_st:first_frm_fn,:])
    return Rzz,Rvv
