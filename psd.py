import os
import joblib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import functions as fn

#------------------------------------------------------------------------------------------------------------

#Legge da terminale il file da leggere

binarized_trajectory = sys.argv[1]

#Leggi il nome della directory da trajectory_file
directory = os.path.dirname(binarized_trajectory)
filename_bin_traj = os.path.basename(binarized_trajectory)

threshold_choice = filename_bin_traj.split('_')[1].split('.')[0]

config = configparser.ConfigParser()
config_file = os.path.join(directory, 'configuration.txt')

config.read(config_file)

omega = float(config['simulation_parameters']['omega'])
h = float(config['simulation_parameters']['h'])

forcing_period = 2 * np.pi / omega
amplitude = float(config['simulation_parameters']['amplitude'])

a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])

expected_freq = 1 / forcing_period

min = fn.positive_min_quartic_potential([a, b])
potential_second_derivative_min = fn.quartic_potential_2derivative(min, [a, b])
potential_second_derivative_max = fn.quartic_potential_2derivative(0, [a, b])

binarized_trajectory = joblib.load(binarized_trajectory)

ts = binarized_trajectory['ts']

PSD_means = {}
for key in binarized_trajectory.keys():
    if key == 'ts':
        continue
    binarized_trajectories = np.array(binarized_trajectory[key])
    psd_list = []
    for traj in binarized_trajectories:
        traj = np.array(traj)
        #Trasformata di Fourier della traiettoria
        traj_fft = np.fft.fft(traj)
        n = traj.size
        h = ts[1] - ts[0]
        freqs = np.fft.fftfreq(n, d=h)
        #Frequenze positive
        positive_freqs = freqs[freqs > 0]
        positive_traj_fft = traj_fft[freqs > 0]
        #Calcolo della Power Spectral Density (PSD)
        power_spectrum = np.abs(positive_traj_fft)**2 / (len(traj) * h)
        psd_list.append(power_spectrum)
    psd_list = np.array(psd_list)
    psd_mean = np.mean(psd_list, axis=0)
    PSD_means[key] = psd_mean
PSD_means['ts'] = positive_freqs
#Rinomino la chiave 'ts' in 'frequenza'
PSD_means['frequenza'] = PSD_means.pop('ts')

# Create a figure and a 2x4 grid of subplots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Media delle PSD')

#Grafico psd
for i, key in enumerate([k for k in PSD_means.keys() if k != 'frequenza']):
    ax = axs[i // 4, i % 4]  # Calculate the position of the subplot
    ax.plot(PSD_means['frequenza'], PSD_means[key], label=key)
    ax.axvline(x=expected_freq, color='r', linestyle='--', label='Frequenza attesa', alpha = 0.1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequenza')
    ax.set_ylabel('PSD')
    ax.set_title(f'PSD {key}')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for the main title
plt.savefig('PSD_means.png')

#Calcolo il SNR per ciascuna riga di psd_mean
SNR_list = []
for key in PSD_means.keys():
    if key == 'frequenza':
        continue
    power_spectrum = PSD_means[key]
    max_index = np.argmin(np.abs(positive_freqs - expected_freq)) #indice più vicino alla frequenca attesa
    peak_value = power_spectrum[max_index]
    #Trovo la base del picco 
    num_neighbors = 10
    neighbors_indices = np.arange(max_index - num_neighbors, 
                                 max_index + num_neighbors + 1)
    
    peak_base = np.mean(power_spectrum[neighbors_indices])

    SNR = 2*peak_value / peak_base
    SNR_log = 10*np.log10(SNR)

    print('Key:', key)
    print(f'SNR: {SNR}')
    SNR_list.append((key, SNR, SNR_log))

#Salvo SNR_list

output_file = f'SNR_{threshold_choice}.pkl'

output_file = os.path.join(directory, output_file)

joblib.dump(SNR_list, output_file)

