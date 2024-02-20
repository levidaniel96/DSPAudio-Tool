
import numpy as np
import soundfile as sf
import argparse
import librosa
import sys
from MVDR_numpy import MVDR
sys.path.append('/home/dsi/levidan2/code/LCMV/code_for_git/create_data_set_RIR/multi_channel_algorithms/')
from utils import ifft_shift_RTFs_numpy
from RTF_estimation.estimate_RTFs import estimate_RTFs


if __name__ == '__main__':
    '''
    This script is the main script for the MVDR algorithm. It gets the input audio file and the parameters and returns the output of the MVDR algorithm.
    note that the input audio file should be a multi-channel audio file and there is a known region of speech and noise in the audio file.
    the STFT parametrs can be changed as well as the number of microphones and the reference microphone.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/dsi/levidan2/code/LCMV/code_for_git/create_data_set_RIR/multi_channel_algorithms/y.wav', help='Path to the input audio file')
    parser.add_argument('--output_path', type=str, default='/home/dsi/levidan2/code/LCMV/code_for_git/create_data_set_RIR/multi_channel_algorithms/', help='Path to the output directory')
    parser.add_argument('--NFFT', type=int, default=2048, help='FFT size')
    parser.add_argument('--wlen', type=int, default=2048, help='Window length')
    parser.add_argument('--n_hop', type=int, default=512, help='Hop size')
    parser.add_argument('--Nl', type=int, default=1024, help='length of the RTF cut from the left')
    parser.add_argument('--Nr', type=int, default=1024, help='length of the RTF cut from the right')
    parser.add_argument('--NUP', type=int, default=None, help='Number of frequency bins')
    parser.add_argument('--overlap', type=int, default=None, help='Overlap size')
    parser.add_argument('--M', type=int, default=5, help='Number of microphones')
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
    if params.overlap is None:
        params.overlap = params.wlen - params.n_hop
    aud, input_fs = sf.read(params.input_path)
    ## change fs in paeams if needed
    if input_fs != params.fs:
            aud = librosa.resample(aud.T, orig_sr=input_fs, target_sr=params.fs)  
            aud = aud.T                
    audio = aud
    Qvv,RTFs=estimate_RTFs(params,audio)

    #%% cut the RTFs to Nl and Nr and save them 
    h_cut=np.zeros([params.Nl+params.Nr,params.M])
    h_cut[:params.Nr,:]=np.real(RTFs[:params.Nr,:])
    h_cut[params.Nr:,:]=np.real(RTFs[params.NFFT-params.Nl:params.NFFT,:])
    
    full_RTFs= ifft_shift_RTFs_numpy(h_cut,params)
    
    ## apply MVDR
    y_n=MVDR(audio,full_RTFs,Qvv,params)
    ## normalize the output
    y_n = y_n / np.max(np.abs(y_n))
    sf.write(params.output_path + 'MVDR_output.wav', y_n, params.fs)
    ## normalize the reference
    ref = audio[:,params.ref_mic] / np.max(np.abs(audio[:,params.ref_mic]))
    sf.write(params.output_path + 'ref_mic.wav', ref, params.fs)


