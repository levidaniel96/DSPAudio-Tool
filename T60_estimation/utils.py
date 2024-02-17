import librosa
import numpy as np
import os
import pickle
def Squeeze_freq(stft):
    squeeze_stft,matrix_stft = [],[]   
    temp_stft = stft 
    hamming4 = np.hamming(4)
    hamming4 = np.expand_dims(hamming4/sum(hamming4),axis = 0)
    for row in temp_stft:    
        if (len(np.asarray(matrix_stft)) == 4):                     # Checks that we have reached the fourth row
            squeeze_stft.append(np.matmul(hamming4,matrix_stft))    # Multiply the matrix in hamming window
            matrix_stft=[row]
        else:
            matrix_stft.append(row)
    squeeze_stft = np.squeeze(squeeze_stft,axis = 1)
    return np.asarray(squeeze_stft)

def prepare_data(WAV_PATH):
    win_len = 512
    overLap = 0.75
    R = int(win_len - win_len * overLap)
    ## if file end with wav then read it
    if WAV_PATH.endswith('.wav'):
        y,fs = librosa.load(WAV_PATH,sr=None, mono=True)
    elif WAV_PATH.endswith('.p'):
        with open(WAV_PATH, "rb") as f:
            mix_without_noise, noisy_signal, speakers_target_rev, speakers_delayed_clean, s_thetas_array = pickle.load(f)
            ## extract the first speaker
            y=mix_without_noise[0]
            ## VAD - Voice Activity Detection
            y = y[np.where(abs(y) > 0.01)]
        fs = 16000
    if fs != 16000:
        y = librosa.resample(y, orig_sr=fs, target_sr=16000)    
    if len(y) < 4 * fs:  
        y = np.pad(y, (0, 4 * fs - len(y)), 'constant')
    Y = np.abs(librosa.stft(y, n_fft=win_len, hop_length=R, window='hamming')+1e-8)
    i = 0
    jump = 0
    while Y.shape[1] > (jump + int(fs * 4 / R) - 1)//2:
        data_y = 20 * np.log10(Y[:, 0 + jump:jump + int(fs * 4 / R)])
        # create a batch of matrices
        data_y = Squeeze_freq(data_y)
        if i == 0:
            data_set = np.expand_dims(data_y, axis=0)
        else:
            if data_y.shape[1]!=data_set.shape[2]:
                break
            data_set = np.concatenate((data_set, np.expand_dims(data_y, axis=0)), axis=0)
        i += 1
        jump = int(np.ceil(i / 2 * fs / R))  # 1/2
    return data_set



def extract_wav_paths(base_dir):
    # Initialize an empty list to store the file paths
    wav_paths = []

    # Recursively traverse the directories and collect WAV file paths
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                wav_paths.append(wav_path)  
    return wav_paths

def extract_p_paths(base_dir):
    # Initialize an empty list to store the file paths
    wav_paths = []

    # Recursively traverse the directories and collect WAV file paths
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.p'):
                wav_path = os.path.join(root, file)
                wav_paths.append(wav_path)  
    return wav_paths