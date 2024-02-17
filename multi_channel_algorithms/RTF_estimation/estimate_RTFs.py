import numpy as np
import scipy.signal as ss
from scipy.io import loadmat
import os
import scipy.io as sio
from multi_channel_algorithms.RTF_estimation.GEVD import GEVD
from multi_channel_algorithms.RTF_estimation.creat_Qvv_Qzz import creat_Qvv_Qzz

def  estimate_RTFs(paths,params):
          
    frame_count = 1 + (y.shape[0] - params.wlen ) // params.n_hop
    Y_STFT_matrix=np.zeros([int(eval(params.NUP)),frame_count,params.M],dtype=complex)

    for m in range(params.M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],params.fs, np.hamming(params.wlen) , nperseg=params.wlen, noverlap=params.wlen-params.n_hop, nfft=params.NFFT,boundary=None,padded=False)[2] 
    Qzz,Qvv=creat_Qvv_Qzz(Y_STFT_matrix,params)
    g=GEVD(Qzz,Qvv,params)
    #%% cut the RTFs to Nl_in and Nr_in and save them 
    h_cut=np.zeros([params.Nl_in+params.Nr_in,params.M])
    h_cut[:params.Nr_in,:]=np.real(g[:params.Nr_in,:])
    h_cut[params.Nr_in:,:]=np.real(g[params.NFFT-params.Nl_in:params.NFFT,:])
    RTFs=h_cut[:,[0,1,3,4]] # concatenate the RTFs to vector 
    RTFs_to_net=RTFs.flatten(order='F')
    train_data={}
    train_data['RTFs_to_net']=RTFs_to_net
    
    if not os.path.exists(paths.test_RTFs_path):
        os.makedirs(paths.test_RTFs_path)
    sio.savemat(paths.test_RTFs_path + 'RTFs_to_net.mat',train_data)
    
    return RTFs_to_net

def __main__():
    aud, input_fs = sf.read(fpath)
    fs = sampling_rate
    if input_fs != fs:
        audio = librosa.resample(aud, input_fs, fs)
    else:
        audio = aud    
    data = loadmat(test_data_path+'test_data.mat')  
    y=np.array(data['y'])  

    estimate_RTFs(paths,params)

if __name__ == '__main__':
    __main__()
        