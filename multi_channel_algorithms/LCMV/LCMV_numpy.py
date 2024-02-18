
def LCMV_RTFs(first_spk,second_spk,y,noise,RTFs,time_esti,params.wlen=4096,nfft=4096):#,estimate_RTF='no',RTFs=None):
    #frame_count=Y_STFT_matrix.shape[2]
    win = np.hamming(params.wlen)
    overlap = int(params.wlen * 3 / 4)
    params.NUP = int(nfft/2)+1
    params.n_hop=params.wlen-overlap

    frame_count = 1 + (y.shape[0] - params.wlen ) // params.n_hop

    Y_STFT_matrix=np.zeros([params.NUP,frame_count,M],dtype=complex)
    First_spk_STFT=np.zeros([params.NUP,frame_count,M],dtype=complex)
    Second_spk_STFT=np.zeros([params.NUP,frame_count,M],dtype=complex)
    N_STFT=np.zeros([params.NUP,frame_count,M],dtype=complex)


    for m in range(M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],fs, win , nperseg=params.wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 
        First_spk_STFT[:,:,m]=ss.stft(first_spk[:,m],fs, win , nperseg=params.wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 
        Second_spk_STFT[:,:,m]=ss.stft(second_spk[:,m],fs, win , nperseg=params.wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2]
        N_STFT[:,:,m]=ss.stft(noise[:,m],fs, win , nperseg=params.wlen, noverlap=overlap, nfft=nfft,boundary=None,padded=False)[2] 

    output_y_stft = np.zeros([params.NUP,2,frame_count], dtype=complex)
    output_first_stft = np.zeros([params.NUP,2,frame_count], dtype=complex)
    output_second_stft = np.zeros([params.NUP,2,frame_count], dtype=complex)
    output_n_stft = np.zeros([params.NUP,2,frame_count], dtype=complex)
    e=1e-6
  
    w=np.zeros([M,2,params.NUP],dtype=complex)

    _,Qvv = creat_Qvv_Qzz(Y_STFT_matrix,time_esti,nfft)
    
    for k in range(params.NUP):
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
