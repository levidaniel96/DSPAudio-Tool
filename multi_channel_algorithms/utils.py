import torch
import numpy as np
import math

def ifft_shift_RTFs(RTFs, device, batch_size, M, wlen, Nr, Nl,ref_Mic=2):
    # Calculate the total length of RTFs
    mic_list = torch.arange(M)
    mic_list = torch.cat((mic_list[ref_Mic:], mic_list[:ref_Mic]))
    
    len_of_RTF = Nl + Nr
    
    # Initialize a tensor for shifted RTFs with zeros
    RTFs_ = torch.zeros((batch_size, wlen, M)).to(device)
    
    # Assign values to the shifted RTFs tensor based on indexing using a for loop
    for mic_index in range(M):
        RTFs_[:, :Nr, mic_list[mic_index]] = RTFs[:, mic_index, len_of_RTF * 0:len_of_RTF * 0 + Nr]
        RTFs_[:, wlen - Nl:, mic_list[mic_index]] = RTFs[:, mic_index, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    
    # Set the value at index (0, ref_Mic) to 1 for the reference microphone
    RTFs_[:, 0, ref_Mic] = 1
        
    return RTFs_

def ifft_shift_RTFs_numpy(RTFs,params):
    
    len_of_RTF = params.Nl + params.Nr  
    # Initialize a tensor for shifted RTFs with zeros
    RTFs_ = np.zeros(( params.wlen, params.M))  
    # Assign values to the shifted RTFs tensor based on indexing
    for mic_index in range(params.M):
        RTFs_[ :params.Nr, mic_index] = RTFs[len_of_RTF * 0:len_of_RTF * 0 + params.Nr, mic_index]
        RTFs_[ params.wlen - params.Nl:, mic_index] = RTFs[len_of_RTF * 1 - params.Nl:len_of_RTF * 1, mic_index]
    # Set the value at index (0, ref_Mic) to 1 for the reference microphone
    
    return RTFs_

def create_Qvv_k_batch(Y_STFT_matrix):
    '''
    calculate Qvv for each batch and each frequency point (k) in the STFT domain 
    Y_STFT_matrix: (batch_size,frame_count,M)
    '''  
    Rvv=torch.bmm(Y_STFT_matrix.permute(0, 2, 1),Y_STFT_matrix.conj())/len(Y_STFT_matrix[0,:,0])
    return Rvv

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
    Rzz=np.zeros([int(params.NUP),params.M,params.M],dtype=complex)
    Rvv=np.zeros([int(params.NUP),params.M,params.M],dtype=complex)
    noise_frm_st = int(math.ceil(params.start_time_noise*params.fs/params.n_hop))
    noise_frm_fn = int(math.floor(params.end_time_noise*params.fs/params.n_hop))
    first_frm_st = int(math.ceil(params.start_time_spk*params.fs/params.n_hop))
    first_frm_fn = int(math.floor(params.end_time_spk*params.fs/params.n_hop)) 
    for k in range(int(params.NUP)):
        Rvv[k,:,:]=Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].T@Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].conj()/len(Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:])
        Rzz[k,:,:]=Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].T@Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].conj()/len(Y_STFT_matrix[k,first_frm_st:first_frm_fn,:])
    return Rzz,Rvv
