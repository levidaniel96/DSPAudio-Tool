import numpy as np
from scipy.linalg import eigh, fractional_matrix_power
from scipy.fftpack import ifft

def GEVD(Qzz,Qvv,params):    
    '''
    This function estimate the Relative Transfer Functions (RTFs) using GEVD algorithm 
    Input: 
        Qzz: noise covariance matrix
        Qvv: speech covariance matrix
        params: parameters  
    Output:
        g: RTFs in time domain 
    ''' 
    a_hat_GEVD=np.zeros([int(eval(params.NUP)),params.M],dtype=complex)  
    G_full=np.zeros([params.NFFT,params.M],dtype=complex)
    for k in range(int(eval(params.NUP))):
        D_,V_ = eigh(Qvv[k,:,:])
        idx=np.flip(np.argsort(D_))
        D=D_[idx]
        V=V_[:,idx]
        D_matrix = np.diag(D, k=0)
        Rv1_2 = V @ fractional_matrix_power(D_matrix,1/2) @V.conj().T
        invRv1_2 = V @ fractional_matrix_power(D_matrix,-1/2) @ V.conj().T # inverse noise matrix     
        # Covariance whitening      
        Ry = invRv1_2@Qzz@invRv1_2.conj().T
        L, U = eigh(Ry[k,:,:])        
        idx=np.argmax(L)       
        temp = Rv1_2 @ U[:,idx]
        a_hat_GEVD[k,:] = temp/temp[params.ref_mic]
        ## remove outliers
    for m in range(params.M):
        ind = np.squeeze(np.array(np.where(abs(a_hat_GEVD[:,m])>3*np.mean(abs(a_hat_GEVD[:,m])))))
        if ind.size==1:
            G_full[params.NFFT//2]=1
        else:
            real = (2*np.random.binomial(1,0.5,len(ind))-1)*np.mean(np.real(a_hat_GEVD[:,m]))
            imag = (2*np.random.binomial(1,0.5,len(ind))-1)*np.mean(np.imag(a_hat_GEVD[:,m]))
            a_hat_GEVD[ind,m]=real+1j*imag
    ## reconstruct the RTFs         
    G_full[:int(eval(params.NUP))]=a_hat_GEVD
    G_full[int(eval(params.NUP)):]=np.flip(a_hat_GEVD[1:int(eval(params.NUP))-1], axis=0).conj()
    G_full[params.NFFT//2]=1
    # inverse fft to get the RTFs in time domain
    g=ifft(G_full[:,:].T).T
    return g