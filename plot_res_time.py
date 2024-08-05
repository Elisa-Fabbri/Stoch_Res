import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import configparser
import sys

# Carica i dati
residence_times_file = './RK_forcing/residence_times_min.pkl'

# Legge i dizionari
residence_times_dict = joblib.load(residence_times_file)

# Legge il file di configurazione con sys.argv
config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)

# Estrae i parametri
D_start = float(config['simulation_parameters']['D_start'])
D_end = float(config['simulation_parameters']['D_end'])
num_Ds = int(config['simulation_parameters']['num_Ds'])

# Definisci i valori di D
D_values = np.linspace(D_start, D_end, num_Ds)

# Parametro omega e calcolo del periodo di forzamento
omega = 0.1
forcing_period = (2 * np.pi) / omega

# Dividi D_values in due gruppi
mid_point = len(D_values) // 2
D_values_group1 = D_values[:mid_point]
D_values_group2 = D_values[mid_point:]

# Funzione per creare istogrammi
def plot_histograms(D_values, title, filename):
    fig, axes = plt.subplots(4, 2, figsize=(13, 10))
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
        #ax.hist(normalized_residence_times, bins=100, alpha=0.5, label=f'D = {round(D_value, 2)}', range=(0, 5))
        ax.hist(normalized_residence_times, bins=100, alpha=0.5, label=f'D = {round(D_value, 2)}', range=(0, 5), density=True)

        # Linea verticale a 1 (corrispondente a T = forcing_period)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

        # Aggiunta di etichette e legenda
        #ax.set_ylabel('Counts')
        ax.set_ylabel('Density')
        ax.legend()

        # Imposta l'etichetta per l'asse x
        ax.set_xlabel(r'Residence times $T/T_{\text{forcing}}$')

    # Titolo generale e aggiustamento dello spaziamento tra i subplot
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle

    # Mostra la figura
    plt.savefig(filename)
    plt.close()

image_directory = './immagini/residence_times'

if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Crea i grafici per i due gruppi di valori di rumore
plot_histograms(D_values_group1, 'Residence times distribution for first 8 D values (normalized by forcing period)', os.path.join(image_directory, 'res_times_group1.png'))
plot_histograms(D_values_group2, 'Residence times distribution for last 8 D values (normalized by forcing period)', os.path.join(image_directory, 'res_times_group2.png'))
