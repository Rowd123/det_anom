import numpy as np
import pywt
import matplotlib.pyplot as plt


def main():


    T = 10
    fs = 1000
    f1 = 1
    f2 = 350
    t = np.linspace(0, T, T*fs, endpoint=False)
    X = 1 + np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    wavelet = 'db4'
    d_A, d_B = pywt.dwt(X,wavelet)

    plt.figure(figsize=(10,10))
    plt.subplot(5,1,1)
    plt.plot(t, X)
    plt.title('Original values')

    N_sam = d_A.shape[0]
    t_sam = np.linspace(0, T, N_sam, endpoint=False)

    plt.subplot(5,1,2)
    plt.plot(t_sam, d_A)
    plt.title('Approximation values')


    plt.subplot(5, 1, 3)
    plt.plot(t_sam, d_B)
    plt.title('Detail values')


    d_A, d_B = pywt.dwt(d_A,wavelet)

    N_sam = d_A.shape[0]
    t_sam = np.linspace(0, T, N_sam, endpoint=False)

    plt.subplot(5,1,4)
    plt.plot(t_sam, d_A)
    plt.title('Approximation values')

    plt.subplot(5, 1, 5)
    plt.plot(t_sam, d_B)
    plt.title('Detail values')

    plt.show()

if __name__ == "__main__":
    main()