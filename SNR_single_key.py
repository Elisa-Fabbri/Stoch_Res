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
SNR_list = []
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
    #Calcolo il SNR 
    max_index = np.argmin(np.abs(positive_freqs - expected_freq))  # indice più vicino alla frequenza attesa
    peak_value = power_spectrum[max_index]
    # Trova la base del picco
    num_neighbors = 5
    neighbors_indices = np.arange(max_index - num_neighbors, max_index + num_neighbors + 1)

    peak_base = np.mean(power_spectrum[neighbors_indices])

    SNR = 2 * peak_value / peak_base
    SNR_list.append(SNR)

SNR_mean = np.mean(SNR_list)
SNR_std = np.std(SNR_list)

#Salva il risultato in un file csv con la chiave e il valore della media e della deviazione standard
#Se il file con lo stesso nome esiste già, aggiunge la riga al file

file_name = os.path.join(directory, 'SNR.csv')
if not os.path.exists(file_name):
    with open(file_name, 'w') as file:
        file.write('key,mean,std\n')

#Se il file esiste già controllo se la chiave è già presente, se non c'è la aggiungo

with open(file_name, 'r') as file:
    lines = file.readlines()
    keys = [line.split(',')[0] for line in lines[1:]]
    if key not in keys:
        with open(file_name, 'a') as file:
            file.write(f'{key},{SNR_mean},{SNR_std}\n')
    else: 
        print(f'La chiave {key} è già presente nel file {file_name}')
        print(f'Riga esistente: {lines[keys.index(key) + 1]}')
        print(f'Nuovo valore: {SNR_mean}, {SNR_std}')
    






    



# Calcolo delle PSD per i valori selezionati di D
for key in selected_keys:
    binarized_trajectories = np.array(binarized_trajectory_04[key])
    psd_list_04 = []

    for traj in binarized_trajectories:
        # Trasformata di Fourier della traiettoria
        traj_fft = np.fft.fft(traj)
        n = traj.size
        freqs = np.fft.fftfreq(n, d=h)

        # Frequenze positive
        positive_freqs = freqs[freqs > 0]
        positive_traj_fft = traj_fft[freqs > 0]

        # Calcolo della Power Spectral Density (PSD)
        power_spectrum = np.abs(positive_traj_fft) ** 2 / (len(traj) * h)
        psd_list_04.append(power_spectrum)

        # Elimina variabili non necessarie
        del traj, traj_fft, power_spectrum

    print(f'Key: {key}', 'PSD calcolate')
    psd_list_04 = np.array(psd_list_04)

    psd_mean_04 = np.mean(psd_list_04, axis=0)

    PSD_means_04[key] = psd_mean_04

PSD_means_04['frequenza'] = positive_freqs

# Elimina variabili non necessarie
del binarized_trajectories, psd_list_04

# Creazione del grafico con 4 subplot per alcuni valori di rumore di PSD_means_04
fig, axs = plt.subplots(2, 3, figsize=(16, 8))

for i, key in enumerate(selected_keys):
    ax = axs[i // 3, i % 3]
    D_value = float(key)
    ax.title.set_text(f'D = {round(D_value, 3)}')
    ax.plot(PSD_means_04['frequenza'], PSD_means_04[key], label=f'D = {round(D_value, 3)}')
    ax.axvline(x=expected_freq, color='r', linestyle='--', label='Frequenza attesa', alpha=0.1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequenza')
    ax.set_ylabel('PSD')
    ax.legend(loc='upper right')  # Posiziona la legenda in alto a destra

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(directory_04, 'PSD_means_04.png'))
plt.close()