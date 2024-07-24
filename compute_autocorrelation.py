import os
import joblib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------

#def compute_autocorrelation(trajectory):
#    trajectory = np.array(trajectory)
#    n = len(trajectory)
#    autocorr_i = np.correlate(trajectory, trajectory, mode='full')
#    autocorr_i = autocorr_i[autocorr_i.size // 2:] / n
#    return autocorr_i

#def autocorrelation(trajectories, n_jobs=-1):
    # Trasforma tutte le traiettorie in array numpy
#    trajectories = [np.array(trajectory) for trajectory in trajectories]
#    for trajectory in trajectories:
#        print(trajectory)
    
    # Filtra le traiettorie vuote
#    valid_trajectories = [trajectory for trajectory in trajectories if len(trajectory) > 0]

    # Parallelizzazione del calcolo dell'autocorrelazione per ciascuna traiettoria valida
#    autocorrelations = Parallel(n_jobs=n_jobs)(delayed(compute_autocorrelation)(trajectory) for trajectory in valid_trajectories)
#    return autocorrelations


def autocorrelation(signal):
    # Convert the signal to a numpy array
    signal = np.array(signal)
    
    # Subtract the mean to make the signal zero-mean
    signal = signal - np.mean(signal)
    
    # Compute the FFT of the zero-padded signal
    fft_signal = np.fft.fft(signal, n=2*len(signal)-1)
    
    # Compute the power spectrum (element-wise multiplication of the FFT and its complex conjugate)
    power_spectrum = fft_signal * np.conj(fft_signal)
    
    # Compute the inverse FFT of the power spectrum
    autocorr = np.fft.ifft(power_spectrum).real
    
    # Normalize the autocorrelation
    autocorr /= len(signal)
    
    # Return only the first part of the result (positive lags)
    return autocorr[:len(signal)]

def fourier_transform(signal, times):
    # Compute the FFT of the signal
    fft_signal = np.fft.fft(signal)
    
    # Compute the frequencies
    freqs = np.fft.fftfreq(len(signal), d=times[1]-times[0])
    
    return freqs, fft_signal

#------------------------------------------------------------------------------------------------------------

#Legge da terminale il file da leggere

binarized_trajectory = sys.argv[1]

#Leggi il nome della directory da trajectory_file
directory = os.path.dirname(binarized_trajectory)
filename = os.path.basename(binarized_trajectory)

threshold_choice = filename.split('_')[1].split('.')[0]

output_file = f'autocorrelation_{threshold_choice}.pkl'
output_file = os.path.join(directory, output_file)

binarized_trajectory = joblib.load(binarized_trajectory)

ts = binarized_trajectory['ts']

for key in binarized_trajectory.keys():
    if key == 'ts':
        continue
    binarized_trajectories = np.array(binarized_trajectory[key])
    autocorr_list = []
    for traj in binarized_trajectories:
        traj = np.array(traj)
        autocorr = np.correlate(traj, traj, mode='full')
        autocorr = autocorr[autocorr.size // 2:] / len(traj)
        autocorr_list.append(autocorr)
    autocorr_list = np.array(autocorr_list)
    autocorr_mean = np.mean(autocorr_list, axis=0)
    binarized_trajectory[key] = autocorr_mean
    print('Sono vivo!')

joblib.dump(binarized_trajectory, output_file)
    

        


