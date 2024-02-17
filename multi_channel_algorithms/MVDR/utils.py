import torch
import numpy as np
def ifft_shift_RTFs(RTFs, device, batch_size, M, wlen, Nr, Nl,ref_Mic=2):
    # Calculate the total length of RTFs
    mic_list = np.arange(M)
    mic_list = np.delete(mic_list, ref_Mic)
    
    len_of_RTF = Nl + Nr
    
    # Initialize a tensor for shifted RTFs with zeros
    RTFs_ = torch.zeros((batch_size, wlen, M)).to(device)
    
    # Assign values to the shifted RTFs tensor based on indexing
    RTFs_[:, :Nr,mic_list[0]] = RTFs[:, 0, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    RTFs_[:, :Nr, mic_list[1]] = RTFs[:, 1, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    RTFs_[:, :Nr, mic_list[2]] = RTFs[:, 2, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    RTFs_[:, :Nr, mic_list[3]] = RTFs[:, 3, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    
    RTFs_[:, wlen - Nl:, mic_list[0]] = RTFs[:, 0, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    RTFs_[:, wlen - Nl:, mic_list[1]] = RTFs[:, 1, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    RTFs_[:, wlen - Nl:, mic_list[2]] = RTFs[:, 2, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    RTFs_[:, wlen - Nl:, mic_list[3]] = RTFs[:, 3, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    
    # Set the value at index (0, ref_Mic) to 1 for the reference microphone
    RTFs_[:, 0, ref_Mic] = 1
        
    return RTFs_

def create_Qvv_k_batch(Y_STFT_matrix):
    '''
    calculate Qvv for each batch and each frequency point (k) in the STFT domain 
    Y_STFT_matrix: (batch_size,frame_count,M)
    '''  
    Rvv=torch.bmm(Y_STFT_matrix.permute(0, 2, 1),Y_STFT_matrix.conj())/len(Y_STFT_matrix[0,:,0])
    return Rvv