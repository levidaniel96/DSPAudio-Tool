
import numpy as np

import scipy.signal as ss
from numpy import linalg as LA
import math
import torch
nfft = 4096
NUP = int(nfft/2)+1

wlen = 4096
fs = 16000
overlap = int(wlen * 3 / 4)
n_hop=wlen-overlap
win = np.hamming(wlen)
M=5
overLap = 0.75
noverLap=overLap*wlen
R = wlen-wlen*overLap

def creat_Qvv_Qzz(Y_STFT_matrix,time_esti,nfft=4096):
    NUP = int(nfft/2)+1

    wlen = nfft#//2
    fs = 16000
    overLap = 0.75
    R = wlen-wlen*overLap

    Rzz=np.zeros([NUP,M,M],dtype=complex)
    Rvv=np.zeros([NUP,M,M],dtype=complex)
    noise_frm_st = int(math.ceil(0*fs/R))
    noise_frm_fn = int(math.floor(time_esti*fs/R-1))
    first_frm_st = int(math.ceil(2*fs/R))
    first_frm_fn = int(math.floor((2+time_esti)*fs/R)) 
    for k in range(NUP):
        Rvv[k,:,:]=Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].T@Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].conj()/len(Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:])
        Rzz[k,:,:]=Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].T@Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].conj()/len(Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:])
    return Rzz,Rvv






def MVDR(x,y,noise,RTFs,time_esti):
    frame_count = 1 + (y.shape[0] - wlen ) // n_hop

    Y_STFT_matrix=np.zeros([NUP,frame_count,M],dtype=complex)
    X_STFT=np.zeros([NUP,frame_count,M],dtype=complex)
    N_STFT=np.zeros([NUP,frame_count,M],dtype=complex)

    for m in range(M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 
        X_STFT[:,:,m]=ss.stft(x[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] ###########
        N_STFT[:,:,m]=ss.stft(noise[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] ##########
    output_y_stft = np.zeros([NUP,frame_count], dtype=complex)
    output_x_stft = np.zeros([NUP,frame_count], dtype=complex)
    output_n_stft = np.zeros([NUP,frame_count], dtype=complex)
    e=7/10000
  
    w=np.zeros([M,NUP],dtype=complex)

    _,Qvv = creat_Qvv_Qzz(Y_STFT_matrix,time_esti)

    for f in range(NUP):
        inv_qvv = LA.inv(np.squeeze(Qvv[f, :, :]) + e * LA.norm(Qvv[f, :, :]) * np.eye(M))      
        b = inv_qvv @ RTFs[f,:]    
        denom = np.squeeze(RTFs[f, :]).conj().T @ b

        w = b / denom
        output_y_stft[f,:]=w.conj() @ np.squeeze(Y_STFT_matrix[f,:,:]).T
        output_x_stft[f,:]=w.conj() @ np.squeeze(X_STFT[f,:,:]).T
        output_n_stft[f,:]=w.conj() @ np.squeeze(N_STFT[f,:,:]).T
        
    return output_y_stft,output_x_stft,output_n_stft


def LCMV_RTFs(first_spk,second_spk,y,noise,RTFs,time_esti,wlen=4096,nfft=4096):#,estimate_RTF='no',RTFs=None):
    #frame_count=Y_STFT_matrix.shape[2]
    win = np.hamming(wlen)
    overlap = int(wlen * 3 / 4)
    NUP = int(nfft/2)+1
    n_hop=wlen-overlap

    frame_count = 1 + (y.shape[0] - wlen ) // n_hop

    Y_STFT_matrix=np.zeros([NUP,frame_count,M],dtype=complex)
    First_spk_STFT=np.zeros([NUP,frame_count,M],dtype=complex)
    Second_spk_STFT=np.zeros([NUP,frame_count,M],dtype=complex)
    N_STFT=np.zeros([NUP,frame_count,M],dtype=complex)


    for m in range(M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 
        First_spk_STFT[:,:,m]=ss.stft(first_spk[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 
        Second_spk_STFT[:,:,m]=ss.stft(second_spk[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2]
        N_STFT[:,:,m]=ss.stft(noise[:,m],fs, win , nperseg=wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 

    output_y_stft = np.zeros([NUP,2,frame_count], dtype=complex)
    output_first_stft = np.zeros([NUP,2,frame_count], dtype=complex)
    output_second_stft = np.zeros([NUP,2,frame_count], dtype=complex)
    output_n_stft = np.zeros([NUP,2,frame_count], dtype=complex)
    e=1e-6
  
    w=np.zeros([M,2,NUP],dtype=complex)

    _,Qvv = creat_Qvv_Qzz(Y_STFT_matrix,time_esti,nfft)
    
    for k in range(NUP):
        g = (RTFs[k,:,:].squeeze())
        b = (Qvv[k,:,:].squeeze())
        inv_b = b + e*np.linalg.norm(b)*np.eye(M)
        c = np.matmul(np.linalg.inv(inv_b),g)
        g_conj = np.conj(g).T
        inv_temp =  np.matmul(g_conj,c) + e*np.linalg.norm(np.matmul(g_conj,c))*np.eye(2)
        w[:,:,k] = (np.matmul(c , np.linalg.inv(inv_temp)))
        
        output_y_stft[k,:,:] = np.matmul(np.conj(w[:,:,k].squeeze()).T,(Y_STFT_matrix[k,:,:].squeeze()).T)
        output_first_stft[k,:,:] = np.matmul(np.conj(w[:,:,k].squeeze()).T,(First_spk_STFT[k,:,:].squeeze()).T)
        output_second_stft[k,:,:] = np.matmul(np.conj(w[:,:,k].squeeze()).T,(Second_spk_STFT[k,:,:].squeeze()).T)
        output_n_stft[k,:,:] = np.matmul(np.conj(w[:,:,k].squeeze()).T,(N_STFT[k,:,:].squeeze()).T)


        
    return output_y_stft,output_first_stft,output_second_stft,output_n_stft









