import torch
from multi_channel_algorithms.utils import ifft_shift_RTFs,create_Qvv_k_batch

def MVDR_noisy_and_oracle_loss(y,RTFs,args,device,batch_size):
    '''
    This function calculates the MVDR loss for a batch of signals
    inputs:
        y: (batch_size,frame_count,M)
        RTFs: (batch_size,M-1,len_of_RTF)
        args: args object
        device: device to use
        batch_size: batch size
    outputs:
        y_hat: (batch_size,frame_count)
    '''
    win = torch.hamming_window(args.wlen).to(device)
    e=1e-6
    eye_M=torch.eye(args.M).repeat(batch_size, 1, 1).to(device)
    frame_count = 1 + (y.shape[1] - args.wlen ) //args.n_hop
 
    h_rtfs=ifft_shift_RTFs(RTFs,device,batch_size,args.M,args.wlen,args.Nr,args.Nl,ref_Mic=args.ref_mic)


    Y_STFT_matrix=torch.zeros((batch_size,int(args.NUP),frame_count,args.M),dtype=torch.cfloat).to(device)
    for m in range(args.M):
        Y_STFT_matrix[:,:,:,m]=torch.stft(y[:,:,m],args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
    output_y_stft = torch.zeros(batch_size,int(args.NUP),frame_count, dtype=torch.cfloat).to(device)
  

    H_1 = torch.fft.fft(h_rtfs,dim=1)
    
    for f in range(int(args.NUP)):
        # calculate Qvv for each batch and each frequency point (k) in the STFT domain 
        Qvv=create_Qvv_k_batch(Y_STFT_matrix[:,f,:frame_count//5,:])
        H_k = torch.unsqueeze(torch.squeeze(H_1[:,f,:]),2)
        # MVDR weights calculation - w 
        inv_qvv = torch.inverse(Qvv+e*torch.norm(Qvv,dim=(1,2))[:, None, None]*eye_M) #+ e * LA.norm(Qvv[f, :, :]) * torch.eye(M).to(device))
        b = torch.bmm(inv_qvv,H_k)
        inv_temp = torch.squeeze(torch.bmm(H_k.conj().permute(0, 2, 1) , b)) + e*torch.norm(torch.bmm(H_k.conj().permute(0, 2, 1) , b),dim=(1,2))
        w =(torch.squeeze(b).T/inv_temp).T
        # calculate output for each batch and each frequency point (k) in the STFT domain
        output_y_stft[:,f,:]=torch.squeeze(torch.bmm(torch.unsqueeze(w.conj(),1) , torch.squeeze(Y_STFT_matrix[:,f,:,:]).permute(0, 2, 1)))
    y_hat=torch.istft(output_y_stft,args.NFFT,args.n_hop,args.wlen,win) 

    return y_hat

