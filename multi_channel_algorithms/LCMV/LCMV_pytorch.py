
import torch
from multi_channel_algorithms.utils import create_Qvv_k_batch

def LCMV_torch(y, h_first_spk, h_second_spk, args, device, batch_size):
    '''
    :param y: noisy mixture signal (batch_size, time, M)
    :param h_first_spk: RTF for first speaker in time domain (batch_size, time, M) 
    :param h_second_spk: RTF for second speaker in time domain (batch_size, time, M)
    :param args: arguments
    
    '''
    # Constants and initialization
    win = torch.hamming_window(args.wlen).to(device)
    e = 1e-6
    eye_M = torch.eye(args.M).repeat(batch_size, 1, 1).to(device)
    eye_2 = torch.eye(2).repeat(batch_size, 1, 1).to(device)
    frame_count = 1 + (y.shape[1] - args.wlen) // args.n_hop
    Y_STFT_matrix = torch.zeros((batch_size, int(eval(args.NUP)), frame_count, args.M), dtype=torch.cfloat).to(device)

    # STFT calculation for each source
    for m in range(args.M):
        Y_STFT_matrix[:, :, :, m] = torch.stft(y[:, :, m], args.NFFT, args.n_hop, args.wlen, win, center=False, return_complex=True)

    output_y_stft = torch.zeros(batch_size, int(eval(args.NUP)), 2, frame_count, dtype=torch.cfloat).to(device)

    # FFT and concatenation for network and oracle sources
    H_0 = torch.unsqueeze(torch.fft.fft(h_first_spk, dim=1), 3)
    H_1 = torch.unsqueeze(torch.fft.fft(h_second_spk, dim=1), 3)
    H = torch.cat((H_0, H_1), dim=3)

    # Processing each frequency bin
    for f in range(int(eval(args.NUP))):
        # Qvv is the covariance matrix of the noise signal - need to check which frames to use 
        Qvv = create_Qvv_k_batch(Y_STFT_matrix[:, f, :args.noise_frame_end, :])
        H_k = torch.squeeze(H[:, f, :, :])

        inv_qvv = torch.inverse(Qvv + e * torch.norm(Qvv, dim=(1, 2))[:, None, None] * eye_M)

        b = torch.bmm(inv_qvv, H_k)
        inv_temp = torch.bmm(H_k.conj().permute(0, 2, 1), b) + e * torch.norm(
            torch.bmm(H_k.conj().permute(0, 2, 1), b), dim=(1, 2))[:, None, None] * eye_2
        w = torch.bmm(b, torch.inverse(inv_temp))
        output_y_stft[:, f, :] = torch.bmm(torch.squeeze(w.conj().permute(0, 2, 1)),
                                               torch.squeeze(Y_STFT_matrix[:, f, :]).permute(0, 2, 1))


    # Inverse STFT to obtain time-domain signals
    y_hat_0 = torch.istft(output_y_stft[:, :, 0, :], args.NFFT, args.n_hop, args.wlen, win)
    y_hat_1 = torch.istft(output_y_stft[:, :, 1, :], args.NFFT, args.n_hop, args.wlen, win)



    return y_hat_0, y_hat_1