import numpy as np
import scipy.signal as ss
import os
import scipy.io as sio
import soundfile as sf
import librosa
import argparse
from GEVD import GEVD
from utils import creat_Qvv_Qzz


def  estimate_RTFs(params,y):
         
    frame_count = 1 + (len(y) - params.wlen) // params.n_hop
    Y_STFT_matrix=np.zeros([int(params.NUP),frame_count,params.M],dtype=complex)

    for m in range(params.M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],params.fs, np.hamming(params.wlen) , nperseg=params.wlen, noverlap=params.wlen-params.n_hop, nfft=params.NFFT,boundary=None,padded=False)[2] 
    Qzz,Qvv=creat_Qvv_Qzz(Y_STFT_matrix,params)
    g=GEVD(Qzz,Qvv,params)
    
    
    return g

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/dsi/levidan2/code/LCMV/code_for_git/create_data_set_RIR/multi_channel_algorithms/y.wav', help='Path to the input audio file')
    parser.add_argument('--output', type=str, default='/home/dsi/levidan2/code/LCMV/code_for_git/create_data_set_RIR/multi_channel_algorithms/RTF_estimation/', help='Path to the output directory')
    parser.add_argument('--NFFT', type=int, default=2048, help='FFT size')
    parser.add_argument('--wlen', type=int, default=2048, help='Window length')
    parser.add_argument('--n_hop', type=int, default=512, help='Hop size')
    parser.add_argument('--Nl', type=int, default=1024, help='length of the RTF cut from the left')
    parser.add_argument('--Nr', type=int, default=1024, help='length of the RTF cut from the right')
    parser.add_argument('--NUP', type=int, default=None, help='Number of frequency bins')
    parser.add_argument('--M', type=int, default=4, help='Number of microphones')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling rate')
    parser.add_argument('--ref_mic', type=int, default=2, help='Reference microphone')
    parser.add_argument('--start_time_spk', type=float, default=2, help="start time of the speech segment in seconds")
    parser.add_argument('--end_time_spk', type=float, default=4, help="end time of the speech segment in seconds")
    parser.add_argument('--start_time_noise', type=float, default=0, help="start time of the noise segment in seconds")
    parser.add_argument('--end_time_noise', type=float, default=2, help="end time of the noise segment in seconds")
    params = parser.parse_args()
# Calculate default value for NUP based on NFFT
    if params.NUP is None:
        params.NUP = params.NFFT // 2 + 1
    aud, input_fs = sf.read(params.input)
    ## change fs in paeams if needed
    if input_fs != params.fs:
            aud = librosa.resample(aud.T, orig_sr=input_fs, target_sr=params.fs)  
            aud = aud.T                
    audio = aud
    RTFs=estimate_RTFs(params,audio)
    #%% cut the RTFs to Nl and Nr and save them 
    h_cut=np.zeros([params.Nl+params.Nr,params.M])
    h_cut[:params.Nr,:]=np.real(RTFs[:params.Nr,:])
    h_cut[params.Nr:,:]=np.real(RTFs[params.NFFT-params.Nl:params.NFFT,:])
    ## normalize the RTFs
    h_cut=h_cut/np.max(np.abs(h_cut))
    sf.write(os.path.join(params.output, 'RTFs.wav'), h_cut, params.fs)
    

if __name__ == '__main__':
    __main__()
        