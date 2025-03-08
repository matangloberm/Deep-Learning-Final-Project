import librosa
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import permutation

import pipeline_stft as pipeline


################### pipeline ###################

def run_pipeline(recording_path, label, data_count,permutation, test_length, noise_power, max_count, augment_enable):
    file_count = 0
    for filename in os.listdir(recording_path):
        if filename.endswith(".mp3"):
            file_count += 1
            file_path = os.path.join(recording_path, filename)
            data_raw, Rs = librosa.load(file_path, sr=None)  # data & symbol rate
            data_arr = np.array(data_raw)
            data_arr = data_arr / (max(abs(data_arr))) #normalize
            #print(max(abs(data_arr)))
            #print('Filename: ', filename)
            #print('Sampling Rate: ', Rs, ' samples/sec')
            #print('Data Shape: ', data_arr.shape)


            cropped_length = 24976
            nperseg = 446
            noverlap = nperseg / 2
            fs = Rs
            num_chunks = len(data_arr) // cropped_length
            data_chunks = []
            stft_results = []

            for ii in range(num_chunks):
                data_chunks.append(data_arr[ii * cropped_length: (ii + 1) * cropped_length])

            for ii in range(len(data_chunks)):
                t_stft, f_cropped, Zxx_final = pipeline.transform_stft(data_chunks[ii], fs, nperseg, noverlap, noise_power, augment_enable)

                #Zxx_final = Zxx_final / np.max(Zxx_final)
                min_value = np.min(Zxx_final)
                max_value = np.max(Zxx_final)

                normalized_Zxx = 255*(Zxx_final - min_value)/(max_value - min_value)
                Zxx_final = normalized_Zxx
                #print(np.shape(Zxx_final))
                if file_count >= (permutation-1)*test_length+1 and file_count <= permutation*test_length:
                    if noise_power == -120:
                        filename_npy = 'data/test_' + str(permutation) + '/' + label + '/' + label + '_' + filename + '_' + str(noise_power) + '_' + str(ii) + '.npy'
                        np.save(filename_npy, Zxx_final)
                else:
                    if data_count < max_count:
                        filename_npy = 'data/training_' + str(permutation) + '/' + label + '/' + label + '_' + filename + '_' + str(noise_power) + '_' +  str(ii) + '.npy'
                        np.save(filename_npy, Zxx_final)
                        data_count += 1

                #plt.figure()
                #plt.pcolormesh(t_stft[1:], f_cropped, Zxx_final, shading='gouraud', cmap='inferno')
                #plt.show()
                #print('test')
                # stft_results.append(Zxx_final)




    print('FINISHED RUNNING PREPROCESSING PIPELINE')
    print('Total Data Matrices Created: ' + str(data_count))
    return data_count

################### run pipeline ##################

noise_list = np.linspace(-50, -120, 13)
# with SNR augmentation: use noise powers of -120, -115, ..., -70 per permutation
for index in range(5):
    max_count = 3000

    for label in ['UAV', 'HELICOPTER', 'QUADCOPTER']:
        recording_path = 'recordings/' + label + '/'
        test_length = 2 # 2 test videos for heli, quadcopter and 3 for UAV due to lack of data
        if (label == 'UAV'):
            test_length += 1

        data_count = 0

        for noise_power in noise_list:
            permutation = index + 1
            data_count = run_pipeline(recording_path, label, data_count,permutation, test_length, noise_power, max_count, augment_enable=1)
            print(data_count)










'''
##################  crop data and generate spectral chunks ################
num_of_samples = 1000
num_of_chunks = len(data_arr) // num_of_samples

fs = Rs # sampling frequency
T = 1/fs
t = np.linspace(0,(num_of_samples-1) * T, num_of_samples)
freq_arr = np.fft.fftfreq(num_of_samples, T)

chunk_arr = []
spectral_chunks = []

for ii in range(num_of_chunks):
    chunk_arr.append(data_arr[ii * num_of_samples : (ii+1) * num_of_samples])


for chunk in chunk_arr:
    spectrum = np.fft.fft(chunk)
    spectral_chunks.append(spectrum)

plt.figure()
plt.plot(freq_arr[:num_of_samples//2], np.abs((spectral_chunks[0])[0:num_of_samples//2]))
plt.plot(freq_arr[:num_of_samples//2], np.abs((spectral_chunks[1])[0:num_of_samples//2]))
plt.plot(freq_arr[:num_of_samples//2], np.abs((spectral_chunks[2])[0:num_of_samples//2]))
plt.plot(freq_arr[:num_of_samples//2], np.abs((spectral_chunks[3])[0:num_of_samples//2]))
plt.show()

'''

