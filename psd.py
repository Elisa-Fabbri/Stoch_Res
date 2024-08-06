import os
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
import configparser
import functions as fn

#------------------------------------------------------------------------------------------------------------

# Legge da terminale il file da leggere
binarized_trajectory = sys.argv[1]

# Leggi il nome della directory da trajectory_file
directory = os.path.dirname(binarized_trajectory)
filename_bin_traj = os.path.basename(binarized_trajectory)

# Estrai la scelta della soglia dal nome del file
threshold_choice = filename_bin_traj.split('_')[1].split('.')[0]

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
binarized_trajectory = joblib.load(binarized_trajectory)
ts = binarized_trajectory['ts']

# Calcolo delle medie PSD
PSD_means = {}
for key in binarized_trajectory.keys():
    if key == 'ts':
        continue
    binarized_trajectories = np.array(binarized_trajectory[key])
    psd_list = []
    for traj in binarized_trajectories:
        traj = np.array(traj)
        # Trasformata di Fourier della traiettoria
        traj_fft = np.fft.fft(traj)
        n = traj.size
        h = ts[1] - ts[0]
        freqs = np.fft.fftfreq(n, d=h)
        # Frequenze positive
        positive_freqs = freqs[freqs > 0]
        positive_traj_fft = traj_fft[freqs > 0]
        # Calcolo della Power Spectral Density (PSD)
        power_spectrum = np.abs(positive_traj_fft)**2 / (len(traj) * h)
        psd_list.append(power_spectrum)
    psd_list = np.array(psd_list)
    psd_mean = np.mean(psd_list, axis=0)
    PSD_means[key] = psd_mean

PSD_means['ts'] = positive_freqs
# Rinomina la chiave 'ts' in 'frequenza'
PSD_means['frequenza'] = PSD_means.pop('ts')

# Separa le chiavi in due gruppi per creare due immagini
keys = [k for k in PSD_means.keys() if k != 'frequenza']
first_half_keys = keys[:8]   # Prime 8 chiavi
second_half_keys = keys[8:]  # Ultime 8 chiavi

def plot_psd(keys_subset, output_filename, title):
    """
    Funzione per plottare un sottoinsieme di chiavi PSD.
    """
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    #fig.suptitle(title)

    for i, key in enumerate(keys_subset):
        ax = axs[i // 4, i % 4]
        D_value = float(key)
        ax.title.set_text(f'D = {round(D_value, 3)}')
        ax.plot(PSD_means['frequenza'], PSD_means[key]) #, label=f'D = {round(D_value, 3)}')
        ax.axvline(x=expected_freq, color='r', linestyle='--', label='Frequenza attesa', alpha=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequenza')
        ax.set_ylabel('PSD')
        ax.legend(loc='upper right')  # Posiziona la legenda in alto a destra

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(directory, output_filename))
    plt.close()

# Crea i plot per le due metà
plot_psd(first_half_keys, 'PSD_means_group1.png', 'Media delle PSD (Primi 8)')
plot_psd(second_half_keys, 'PSD_means_group2.png', 'Media delle PSD (Ultimi 8)')

# Calcolo il SNR per ciascuna riga di psd_mean
SNR_list = []
for key in PSD_means.keys():
    if key == 'frequenza':
        continue
    power_spectrum = PSD_means[key]
    max_index = np.argmin(np.abs(positive_freqs - expected_freq))  # indice più vicino alla frequenza attesa
    peak_value = power_spectrum[max_index]
    # Trova la base del picco
    num_neighbors = 10
    neighbors_indices = np.arange(max_index - num_neighbors, max_index + num_neighbors + 1)

    peak_base = np.mean(power_spectrum[neighbors_indices])

    SNR = 2 * peak_value / peak_base
    #SNR_log = 10 * np.log10(SNR)

    print('Key:', float(key))
    print(f'SNR: {SNR}')
    SNR_list.append((key, SNR)) #, SNR_log))

# Salva SNR_list
output_file = f'SNR_{threshold_choice}.pkl'
output_file = os.path.join(directory, output_file)
joblib.dump(SNR_list, output_file)

# Plotto SNR in funzione di D

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([float(key) for key, _ in SNR_list], [SNR for _, SNR in SNR_list], 'o')
ax.set_xlabel('D')
ax.set_ylabel('SNR')

plt.savefig(os.path.join(directory, f'SNR_{threshold_choice}.png'))
#plt.show()
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([float(key) for key, _ in SNR_list], [SNR for _, SNR in SNR_list], 'o')
ax.set_xlabel('D')
ax.set_ylabel('SNR')
ax.set_yscale('log')
plt.savefig(os.path.join(directory, f'SNR_{threshold_choice}_log.png'))
#plt.show()
plt.close()
