import numpy as np
import pywt

def Meig(dfdata):
    data = dfdata
    waveletname = 'db4'
    cA_1, cD_1 = pywt.dwt(data, waveletname)
    cA_2, cD_2 = pywt.dwt(cA_1, waveletname)
    cA_3, cD_3 = pywt.dwt(cA_2, waveletname)
    cD_3 = np.pad(cD_3, (0, cD_1.size - cD_3.size), mode='constant', constant_values=0)
    cD_2 = np.pad(cD_2, (0, cD_1.size - cD_2.size), mode='constant', constant_values=0)
    cA_3 = np.pad(cA_3, (0, cD_1.size - cA_3.size), mode='constant', constant_values=0)
    X= np.c_[cD_1.T, cD_2.T, cD_3.T, cA_3.T]
    X_cov = np.cov(X.T)
    eig_val_cov, _  = np.linalg.eig(X_cov)
    return eig_val_cov[:2]

def MultiVarPCA(dfdata):
    X = dfdata
    X_cov = np.cov(X.T)
    eig_val_cov, _ = np.linalg.eig(X_cov)
    return eig_val_cov[:2]

def Meig_Zero(dfdata):
    data = dfdata
    waveletname = 'db4'
    cA_1, cD_1 = pywt.dwt(data, waveletname)
    cA_2, cD_2 = pywt.dwt(cA_1, waveletname)
    cA_3, cD_3 = pywt.dwt(cA_2, waveletname)
    cA_4, cD_4 = pywt.dwt(cA_3, waveletname)
    cD_4 = np.pad(cD_4, (0, cD_1.size - cD_4.size), mode='constant', constant_values=0)
    cD_3 = np.pad(cD_3, (0, cD_1.size - cD_3.size), mode='constant', constant_values=0)
    cD_2 = np.pad(cD_2, (0, cD_1.size - cD_2.size), mode='constant', constant_values=0)


    X= np.c_[cD_1.T, cD_2.T, cD_3.T, cD_4.T]

    X_cov = np.cov(X.T)
    eig_val_cov, _ = np.linalg.eig(X_cov)
    return eig_val_cov[:2]