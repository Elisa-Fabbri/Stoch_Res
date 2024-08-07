import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import configparser
import functions as fn
import sys

#------------------------------------------------------------------------------------------------------------

#Legge da terminale il path del file binarizzato

binarized_trajectory_path = sys.argv[1]

directory = os.path.dirname(binarized_trajectory_path)

# Configurazione
config = configparser.ConfigParser()
config_file = os.path.join(directory, 'configuration.txt')
config.read(config_file)

# Parametri dal file di configurazione
omega = float(config['simulation_parameters']['omega'])
h = float(config['simulation_parameters']['h'])
forcing_period = 2 * np.pi / omega
amplitude = float(config['simulation_parameters']['amplitude'])
a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])

# Frequenza attesa
expected_freq = 1 / forcing_period

# Calcolo dei derivati e del minimo potenziale
min_potential = fn.positive_min_quartic_potential([a, b])
potential_second_derivative_min = fn.quartic_potential_2derivative(min_potential, [a, b])
potential_second_derivative_max = fn.quartic_potential_2derivative(0, [a, b])

# Carica la traiettoria binarizzata
binarized_trajectory = joblib.load(binarized_trajectory_path)

ts = binarized_trajectory['ts']

##########################################################################################

#Legge da terminale la chiave per cui calcolare il SNR

key_index = sys.argv[2]

# Selezione delle chiavi specifiche per i grafici
keys = [k for k in binarized_trajectory.keys() if k != 'ts']
keys = sorted(keys, key=lambda x: float(x))

key = keys[int(key_index)]

binarized_trajectories = binarized_trajectory[key]
PSD_list = []
for traj in binarized_trajectories:
    if len(traj) != len(ts):
        raise ValueError(f'La traiettoria {key} ha una lunghezza diversa da ts')
    traj_fft = np.fft.fft(traj)
    n = traj.size
    freqs = np.fft.fftfreq(n, d=h)

    # Frequenze positive
    positive_freqs = freqs[freqs > 0]
    positive_traj_fft = traj_fft[freqs > 0]

    # Calcolo della Power Spectral Density (PSD)
    power_spectrum = np.abs(positive_traj_fft) ** 2 / (len(traj) * h)
    PSD_list.append(power_spectrum)

PSD_list = np.array(PSD_list)
PSD_means = np.mean(PSD_list, axis=0)

#Calcolo il SNR 
max_index = np.argmin(np.abs(positive_freqs - expected_freq))  # indice più vicino alla frequenza attesa
peak_value = PSD_means[max_index]

# Trova la base del picco
num_neighbors = 5
neighbors_indices = np.arange(max_index - num_neighbors, max_index + num_neighbors + 1)
peak_base = np.mean(PSD_means[neighbors_indices])

SNR = 2 * peak_value / peak_base

#Salva il risultato in un file csv con la chiave e il valore del SNR

file_name = os.path.join(directory, 'SNR.csv')
if not os.path.exists(file_name):
    with open(file_name, 'w') as file:
        file.write('key,SNR\n')

#Se il file esiste già controllo se la chiave è già presente, se non c'è la aggiungo

with open(file_name, 'r') as file:
    lines = file.readlines()
    keys = [line.split(',')[0] for line in lines[1:]]
    if key not in keys:
        with open(file_name, 'a') as file:
            file.write(f'{key},{SNR}\n')
    else: 
        print(f'La chiave {key} è già presente nel file {file_name}')
        print(f'Riga esistente: {lines[keys.index(key) + 1]}')
        print(f'Nuovo valore: {SNR}')