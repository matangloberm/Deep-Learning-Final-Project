from scipy.signal import stft
import numpy as np

def transform_stft(data, fs, nperseg=894, noverlap=447, N0=0, augment_enable=0):
    '''

    :param data: vector - input of stft
    :param fs: sampling rate
    :param nperseg: number of samples per stft
    :param noverlap: number of overlapped samples = 0.5*nperseg
    :return: Zxx_final - logarithmic scaled STFT of data, trimmed at 12kHz
    '''
    f, t_stft, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    noise_dB = N0
    noise_power = 10 ** (noise_dB / 10)
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), np.shape(Zxx))
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), np.shape(Zxx))
    noise = noise_real + 1j * noise_imag
    Zxx = Zxx + noise * augment_enable #when applying augmentation add AWGN
    Zxx_dB = 20 * np.log10(np.abs(Zxx) + 1e-6) # logarithmic view, add 1e-6 to avoid log(0)

    f_cropped = f[0:len(f) // 2]
    Zxx_dB_cropped = Zxx_dB[0:len(Zxx_dB) // 2] # trimming (most energy is between [0,12kHz])
    Zxx_final = []
    for ii in range(len(Zxx_dB_cropped)):
        Zxx_final.append(Zxx_dB_cropped[ii][1:]) # trim first time sample for dim(Zxx_final) to be 244x244

    return t_stft,f_cropped,Zxx_final