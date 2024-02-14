# Import necessary libraries
from scipy.stats import randint 
import glob
import soundfile as sf   
import os, random
import scipy.io as sio
import numpy as np
import random 
import rir_generator as RG
import sys 
import csv
from utils import generate_room_parameters, extract_wav_paths             

def create_MyDataset(paths,params,flags):
    
    
    csv_file = paths.save_data_set_path + 'parameters.csv'
    # Create directories if they don't exist
    save_dir_data = paths.save_data_set_path + 'data/'
    if not os.path.exists(save_dir_data) and flags.save_wav:
        os.makedirs(save_dir_data)
    save_dir_RIR = paths.save_data_set_path + 'RIR/'
    if not os.path.exists(save_dir_RIR) and flags.save_RIR:
        os.makedirs(save_dir_RIR)    



    # Write room parameters to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = generate_room_parameters(params).keys()
        existing_keys_dict = dict.fromkeys(header)
        existing_keys_dict['fs'] = None  
        existing_keys_dict['Sample_num']=None

        writer.writerow(existing_keys_dict.keys())
        
    
        # Write multiple samples
        for sample_num in range(params.num_of_samples):
            room_params = generate_room_parameters(params)
            room_params['fs'] = params.fs
            room_params['Sample_num'] = sample_num
            
            writer.writerow(room_params.values())    
            
            # Generate room impulse response (RIR) for M microphones
            for spk in range(params.num_spk):
                h = RG.generate(
                    c=340,  # Sound velocity (m/s)
                    fs=params.fs,  # Sample frequency (samples/s)
                    r = [
                        [room_params['mic_x'] + params.mic_dis_x * (m - params.M / 2), room_params['mic_y'] + params.mic_dis_y * (m - params.M / 2), room_params['mic_z']]
                        for m in range(params.M)
                    ],  # Receiver position(s) [x y z] (m)
                    s=[room_params['sources'][spk]['x'], room_params['sources'][spk]['y'], room_params['sources'][spk]['z']],
                    L=[room_params['Lx'], room_params['Ly'], room_params['Lz']],
                    reverberation_time=room_params['T60'],
                    nsample=int(room_params['T60'] * params.fs),
                )

                # Signal processing for M microphones           
                #%% create noises
                if flags.white_noise:
                    w_n=np.random.randn(params.M,params.record_time*params.fs)
                if flags.env_noise:
                    # rand noise from noise folder
                    noise_path='/dsi/gannot-lab1/users/Daniel_Levi/T60_project/Noisex-92/'
                    noise_paths=extract_wav_paths(noise_path)
                    # rand noise from noise paths
                    rand_num=randint.rvs(0,len(noise_paths))
                    n,sampale_rate=sf.read(noise_paths[rand_num])
                    
                    rand_start_time=randint.rvs(0,len(n)-params.record_time*params.fs)
                    n=n[rand_start_time:rand_start_time+params.record_time*params.fs]
                    n =np.tile(n, (params.M, 1)) 
                Librti_path=paths.clean_data_set_path
                book_folder=random.choice(os.listdir(Librti_path))    
        
                while book_folder=='hist.png':
                    book_folder=random.choice(os.listdir(Librti_path))    
                Librti_path=Librti_path+book_folder+'/'
                book_folder=random.choice(os.listdir(Librti_path))
                Librti_path=Librti_path+book_folder   
                
                
                # Extract only wav files
                wavs = []
                for filename in glob.glob(Librti_path+'/*.wav'):
                    d,sampale_rate=sf.read(filename)
                    wavs.append(d)
                
                wav_num=randint.rvs(0,len(wavs))   
                s=wavs[wav_num] 
                while len(s)<params.record_time*params.fs:
                    wav_num=randint.rvs(0,len(wavs))   
                    s=np.concatenate((s,wavs[wav_num]))
                s=s[0:params.record_time*params.fs]
                x=np.zeros((params.M,len(s)))
                for m in range(params.M):
                    x[m,:]=np.convolve(s, h[:,m], mode='same')
                #%% add noise to the signal 
                SNR_white_noise=randint.rvs(params.SNR_white_noise_low,params.SNR_white_noise_high)
                SNR_env_noise=randint.rvs(params.SNR_env_noise_low,params.SNR_env_noise_high)
                if flags.white_noise and not flags.env_noise:
                    G2=sum(x[params.ref_mic,:]**2)/sum(w_n[params.ref_mic,:]**2)*10**(-SNR_white_noise/10)
                    noise=np.sqrt(G2)*w_n
                    y=x+noise
                elif flags.env_noise and not flags.white_noise:
                    G2=sum(x[params.ref_mic,:]**2)/sum(n[params.ref_mic,:]**2)*10**(-SNR_env_noise/10)
                    noise=np.sqrt(G2)*n
                    y=x+noise
                elif flags.white_noise and flags.env_noise:
                    G2=sum(x[params.ref_mic,:]**2)/sum(n[params.ref_mic,:]**2)*10**(-SNR_env_noise/10)
                    noise=np.sqrt(G2)*n
                    G2=sum(x[params.ref_mic,:]**2)/sum(w_n[params.ref_mic,:]**2)*10**(-SNR_white_noise/10)
                    noise2=np.sqrt(G2)*w_n
                    noise=noise+noise2
                    y=x+noise
                else:   
                    y=x
                ## normalize y           
                y=y/np.max(np.abs(y))

                #%% save data as wav files
                if flags.save_wav:
                    sf.write(save_dir_data+'y_'+str(sample_num)+'_spk_'+str(spk)+'.wav', y.T, params.fs)
                if flags.save_RIR:
                    sf.write(save_dir_RIR+'h_'+str(sample_num)+'_spk_'+str(spk)+'.wav', h, params.fs)