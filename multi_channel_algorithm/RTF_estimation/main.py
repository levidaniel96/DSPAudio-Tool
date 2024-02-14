
def  estimate_RTFs(paths,params):
    data = loadmat(paths.test_data_path+'test_data.mat')  
    y=np.array(data['y'])        
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