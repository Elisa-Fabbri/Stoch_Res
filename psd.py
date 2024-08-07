import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import configparser
import functions as fn

#------------------------------------------------------------------------------------------------------------

binarized_trajectory_04_path = './RK_forcing_0.4/binarized_min.pkl'
binarized_trajectory_02_path = './RK_forcing_0.2/binarized_min.pkl'

directory_04 = os.path.dirname(binarized_trajectory_04_path)
directory_02 = os.path.dirname(binarized_trajectory_02_path)

filename_bin_traj_04 = os.path.basename(binarized_trajectory_04_path)
filename_bin_traj_02 = os.path.basename(binarized_trajectory_02_path)

# Estrai la scelta della soglia dal nome del file
#threshold_choice_04 = filename_bin_traj_04.split('_')[2].split('.')[0]
#threshold_choice_02 = filename_bin_traj_02.split('_')[2].split('.')[0]


# Configurazione
config_04 = configparser.ConfigParser()
config_file_04 = os.path.join(directory_04, 'configuration.txt')
config_04.read(config_file_04)

config_02 = configparser.ConfigParser()
config_file_02 = os.path.join(directory_02, 'configuration.txt')
config_02.read(config_file_02)

# Parametri dal file di configurazione

omega_04 = float(config_04['simulation_parameters']['omega'])
h_04 = float(config_04['simulation_parameters']['h'])
forcing_period_04 = 2 * np.pi / omega_04
amplitude_04 = float(config_04['simulation_parameters']['amplitude'])
a_04 = float(config_04['potential_parameters']['a'])
b_04 = float(config_04['potential_parameters']['b'])

omega_02 = float(config_02['simulation_parameters']['omega'])
h_02 = float(config_02['simulation_parameters']['h'])
forcing_period_02 = 2 * np.pi / omega_02
amplitude_02 = float(config_02['simulation_parameters']['amplitude'])
a_02 = float(config_02['potential_parameters']['a'])
b_02 = float(config_02['potential_parameters']['b'])

assert omega_04 == omega_02
assert h_04 == h_02
assert amplitude_04 != amplitude_02
assert a_04 == a_02
assert b_04 == b_02

# Rinomina i valori uguali

omega = omega_04
h = h_04
forcing_period = forcing_period_04
a = a_04
b = b_04

# Frequenza attesa
expected_freq = 1 / forcing_period

# Calcolo dei derivati e del minimo potenziale
min_potential = fn.positive_min_quartic_potential([a, b])
potential_second_derivative_min = fn.quartic_potential_2derivative(min_potential, [a, b])
potential_second_derivative_max = fn.quartic_potential_2derivative(0, [a, b])

# Carica la traiettoria binarizzata
binarized_trajectory_02 = joblib.load(binarized_trajectory_02_path)
binarized_trajectory_04 = joblib.load(binarized_trajectory_04_path)

ts_02 = binarized_trajectory_02['ts']
ts_04 = binarized_trajectory_04['ts']

assert np.array_equal(ts_02, ts_04)

ts = ts_02

# Calcolo delle medie PSD
PSD_means_02 = {}
PSD_means_04 = {}

for key_02, key_04 in zip(binarized_trajectory_02.keys(), binarized_trajectory_04.keys()):
    assert key_02 == key_04
    key = key_02
    if key == 'ts':
        continue

    binarized_trajectories_02 = np.array(binarized_trajectory_02[key])
    binarized_trajectories_04 = np.array(binarized_trajectory_04[key])
    psd_list_02 = []
    psd_list_04 = []

    # Process each trajectory in chunks
    for i in range(len(binarized_trajectories_02)):
        # Process each trajectory in chunks
        traj_02 = np.array(binarized_trajectories_02[i])
        traj_04 = np.array(binarized_trajectories_04[i])

        # Trasformata di Fourier della traiettoria
        traj_fft_02 = np.fft.fft(traj_02)
        traj_fft_04 = np.fft.fft(traj_04)
        n = traj_02.size
        h = ts[1] - ts[0]
        freqs = np.fft.fftfreq(n, d=h)

        # Frequenze positive
        positive_freqs = freqs[freqs > 0]
        positive_traj_fft_02 = traj_fft_02[freqs > 0]
        positive_traj_fft_04 = traj_fft_04[freqs > 0]

        # Calcolo della Power Spectral Density (PSD)
        power_spectrum_02 = np.abs(positive_traj_fft_02) ** 2 / (len(traj_02) * h)
        power_spectrum_04 = np.abs(positive_traj_fft_04) ** 2 / (len(traj_04) * h)

        psd_list_02.append(power_spectrum_02)
        psd_list_04.append(power_spectrum_04)

        # Elimina variabili non necessarie
        del traj_02, traj_04, traj_fft_02, traj_fft_04, power_spectrum_02, power_spectrum_04

    psd_list_02 = np.array(psd_list_02)
    psd_list_04 = np.array(psd_list_04)

    psd_mean_02 = np.mean(psd_list_02, axis=0)
    psd_mean_04 = np.mean(psd_list_04, axis=0)

    PSD_means_02[key] = psd_mean_02
    PSD_means_02['ts'] = positive_freqs

    PSD_means_04[key] = psd_mean_04
    PSD_means_04['ts'] = positive_freqs

# Elimina variabili non necessarie
del binarized_trajectories_02, binarized_trajectories_04, psd_list_02, psd_list_04

# Rinomina la chiave 'ts' in 'frequenza'

PSD_means_02['frequenza'] = PSD_means_02.pop('ts')
PSD_means_04['frequenza'] = PSD_means_04.pop('ts')

# Creo un grafico con 4 subplot per i alcuni valori di rumore di PSD_means_04 (valori corrispondenti agli 
# indici 1, 2, 3, 5, 5, 10 di D, dove D sono le chiavi di PSD ordinati in ordine crescente)

keys = [k for k in PSD_means_04.keys() if k != 'frequenza']
keys = sorted(keys, key=lambda x: float(x))

keys_subset = [keys[i] for i in [1, 2, 3, 5, 5, 10]]

fig, axs = plt.subplots(2, 3, figsize=(16, 8))

for i, key in enumerate(keys_subset):
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

"""

# Separa le chiavi in due gruppi per creare due immagini
keys = [k for k in PSD_means.keys() if k != 'frequenza']
first_half_keys = keys[:8]   # Prime 8 chiavi
second_half_keys = keys[8:]  # Ultime 8 chiavi

def plot_psd(keys_subset, output_filename, title):
    
    #Funzione per plottare un sottoinsieme di chiavi PSD.
    
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
    
    """