import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import configparser

# Carica i dati
residence_times_file_1 = './RK_forcing_0.4/residence_times_min.pkl'
residence_times_file_2 = './RK_forcing_0.2/residence_times_min.pkl'

# Legge i dizionari
residence_times_dict_1 = joblib.load(residence_times_file_1)
residence_times_dict_2 = joblib.load(residence_times_file_2)

directory1 = os.path.dirname(residence_times_file_1)
directory2 = os.path.dirname(residence_times_file_2)

configuration_file1 = os.path.join(directory1, 'configuration.txt')
configuration_file2 = os.path.join(directory2, 'configuration.txt')

config1 = configparser.ConfigParser()
config1.read(configuration_file1)

config2 = configparser.ConfigParser()
config2.read(configuration_file2)

D_start1 = float(config1['simulation_parameters']['D_start'])
D_end1 = float(config1['simulation_parameters']['D_end'])
num_Ds1 = int(config1['simulation_parameters']['num_Ds'])

D_start2 = float(config2['simulation_parameters']['D_start'])
D_end2 = float(config2['simulation_parameters']['D_end'])
num_Ds2 = int(config2['simulation_parameters']['num_Ds'])

D_values1 = np.linspace(D_start1, D_end1, num_Ds1)
D_values2 = np.linspace(D_start2, D_end2, num_Ds2)

omega = float(config1['simulation_parameters']['omega'])
forcing_period = (2 * np.pi) / omega

# Raggruppa i valori di D in 3 gruppi da 6

D_values_group1 = D_values1[:6]
D_values_group2 = D_values1[6:12]
D_values_group3 = D_values1[12:]

D_values_group4 = D_values2[:6]
D_values_group5 = D_values2[6:12]
D_values_group6 = D_values2[12:]


# Funzione per creare istogrammi
def plot_histograms(D_values, title, filename, residence_times_dict):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for i, D_value in enumerate(D_values):
        row = i // 2  # Riga dell'i-esimo subplot
        col = i % 2   # Colonna dell'i-esimo subplot

        # Ottieni la lista dei tempi di residenza per D_value
        residence_times = residence_times_dict.get(D_value, [])
        if len(residence_times) == 0:
            continue

        # Calcola T/forcing_period per ciascun tempo di residenza
        normalized_residence_times = np.array(residence_times) / forcing_period

        # Plot dell'istogramma nell'i-esimo subplot, limitando l'intervallo dei bin tra 0 e 5
        ax = axes[row, col]
        ax.hist(normalized_residence_times, bins=100, alpha=0.5, label=f'D = {round(D_value, 3)}', range=(0, 5))
        #ax.hist(normalized_residence_times, bins=100, alpha=0.5, label=f'D = {round(D_value, 3)}', range=(0, 5), density=True)

        # Linea verticale a 1 (corrispondente a T = forcing_period)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

        # Aggiunta di etichette e legenda
        ax.set_ylabel('Counts')
        #ax.set_ylabel('Density')
        ax.legend(loc = 'upper right')

        # Imposta l'etichetta per l'asse x
        ax.set_xlabel(r'Residence times $T/T_{\text{forcing}}$')

    # Titolo generale e aggiustamento dello spaziamento tra i subplot
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle

    # Mostra la figura
    plt.savefig(filename)
    plt.close()

image_directory1 = os.path.join(directory1, 'images')
image_directory2 = os.path.join(directory2, 'images')

# Crea la directory se non esiste
if not os.path.exists(image_directory1):
    os.makedirs(image_directory1)

if not os.path.exists(image_directory2):
    os.makedirs(image_directory2)

plot_histograms(D_values_group1, 'Residence times distribution for first 6 D values (normalized by forcing period)', os.path.join(image_directory1, 'res_times_group1.png'), residence_times_dict_1)
plot_histograms(D_values_group2, 'Residence times distribution for next 6 D values (normalized by forcing period)', os.path.join(image_directory1, 'res_times_group2.png'), residence_times_dict_1)
plot_histograms(D_values_group3, 'Residence times distribution for last 6 D values (normalized by forcing period)', os.path.join(image_directory1, 'res_times_group3.png'), residence_times_dict_1)

plot_histograms(D_values_group4, 'Residence times distribution for first 6 D values (normalized by forcing period)', os.path.join(image_directory2, 'res_times_group1.png'), residence_times_dict_2)
plot_histograms(D_values_group5, 'Residence times distribution for next 6 D values (normalized by forcing period)', os.path.join(image_directory2, 'res_times_group2.png'), residence_times_dict_2)
plot_histograms(D_values_group6, 'Residence times distribution for last 6 D values (normalized by forcing period)', os.path.join(image_directory2, 'res_times_group3.png'), residence_times_dict_2)