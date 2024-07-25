"""
Questo modulo plotta l'istogramma dei tempi di residenza
"""
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import configparser

import numpy as np
import matplotlib.pyplot as plt
import joblib

# Carica i dati
residence_times = './Euler_forcing/residence_times_min.pkl'

# Legge i dizionari
Euler_residence_times_dict = joblib.load(residence_times)

D_start = 0.04
D_end = 0.07
num_Ds = 4

# Definisci i valori di D
D_values = np.linspace(0.04, 0.07, 4)

# Crea figure e assi per i subplot
fig, axes = plt.subplots(2, 2, figsize=(13, 6))

# Parametro omega e calcolo del periodo di forzamento
omega = 0.1
forcing_period = (2 * np.pi) / omega

# Itera sui valori di D
for i, D_value in enumerate(D_values):
    row = i // 2  # Riga dell'i-esimo subplot
    col = i % 2   # Colonna dell'i-esimo subplot

    # Ottieni la lista dei tempi di residenza per D_value
    Euler_residence_times = Euler_residence_times_dict.get(D_value, [])
    if len(Euler_residence_times) == 0:
        continue

    # Calcola T/forcing_period per ciascun tempo di residenza
    normalized_residence_times = np.array(Euler_residence_times) / forcing_period

    # Plot dell'istogramma nell'i-esimo subplot, limitando l'intervallo dei bin tra 0 e 5
    ax = axes[row, col]
    ax.hist(normalized_residence_times, bins=200, alpha=0.5, label=f'D = {round(D_value, 2)}', range=(0, 5)) #density = True

    # Linea verticale a 1 (corrispondente a T = forcing_period)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

    # Aggiunta di etichette e legenda
    ax.set_ylabel('Density')
    ax.legend()

    # Imposta l'etichetta per l'asse x
    ax.set_xlabel(r'Residence times $T/T_{\text{forcing}}$')

# Titolo generale e aggiustamento dello spaziamento tra i subplot
plt.suptitle('Residence times distribution for increasing D values (normalized by forcing period)')
plt.tight_layout()

# Mostra la figura
plt.savefig('./Euler_forcing/res_times.png')





"""

#---------------------------------------------------------------------------------------------------------------

def theoretical_residence_time(rk, time):
    return rk * np.exp(-rk * time)


#Legge il file residence_times_flex.pkl da Euler_no_forcing e RK_no_forcing

Euler_folder = './Euler_no_forcing/'
RK_folder = './RK_no_forcing/'

Euler_file_flex = 'residence_times_flex.pkl'
RK_file_flex = 'residence_times_flex.pkl'

Euler_file_min = 'residence_times_min.pkl'
RK_file_min = 'residence_times_min.pkl'

Euler_path_flex = os.path.join(Euler_folder, Euler_file_flex)
RK_path_flex = os.path.join(RK_folder, RK_file_flex)

Euler_path_min = os.path.join(Euler_folder, Euler_file_min)
RK_path_min = os.path.join(RK_folder, RK_file_min)

#Legge i dizionari

Euler_residence_times_dict = joblib.load(Euler_path_flex)
RK_residence_times_dict = joblib.load(RK_path_flex)

#D value va scelto tra i valori di D presenti nei dizionari
#D_values = list(Euler_residence_times_dict.keys())
#print(D_values)

D_value = 0.1388888888888889

#Legge i tempi di residenza
Euler_residence_times = Euler_residence_times_dict[D_value]
RK_residence_times = RK_residence_times_dict[D_value]

# Controllo che i parametri siano uguali per i due metodi

config_file_Euler = './Euler_no_forcing/configuration.txt'
config_file_RK = './RK_no_forcing/configuration.txt'

config_Euler = configparser.ConfigParser()
config_RK = configparser.ConfigParser()

config_Euler.read(config_file_Euler)
config_RK.read(config_file_RK)

a_Euler = float(config_Euler['potential_parameters']['a'])
b_Euler = float(config_Euler['potential_parameters']['b'])

omega_Euler = float(config_Euler['simulation_parameters']['omega'])

a_RK = float(config_RK['potential_parameters']['a'])
b_RK = float(config_RK['potential_parameters']['b'])

omega_RK = float(config_RK['simulation_parameters']['omega'])

assert a_Euler == a_RK
assert b_Euler == b_RK
assert omega_Euler == omega_RK

forcing_period = 2 * np.pi / omega_Euler

min_potential = fn.positive_min_quartic_potential([a_Euler, b_Euler])
second_der_min = fn.quartic_potential_2derivative(min_potential, [a_Euler, b_Euler])
second_der_max = fn.quartic_potential_2derivative(0, [a_Euler, b_Euler])
barrier_height = fn.potential(0, a_Euler, b_Euler) - fn.potential(min_potential, a_Euler, b_Euler)

bin_min = min(min(Euler_residence_times), min(RK_residence_times))
bin_max = max(max(Euler_residence_times), max(RK_residence_times))
bins = np.linspace(bin_min, bin_max, 100)

kr_teorico = fn.kramer_rate(second_der_min, second_der_max, barrier_height, D_value)
theoretical_residence_times_distribution = [theoretical_residence_time(kr_teorico, bin) for bin in bins]

fig = plt.figure()
plt.hist(Euler_residence_times, bins=bins, alpha=0.5, label='Euler', density=True)
plt.hist(RK_residence_times, bins=bins, alpha=0.5, label='RK', density =True)
plt.plot(bins, theoretical_residence_times_distribution, label='Teorico')
plt.legend()
plt.savefig('immagini/residence_time_dist_flex.png')

#Plotto gli istogrammi dei tempi di residenza per diversi valori del rumore e forzante periodica
#Legge il file di residence times di RK_forcing

RK_forcing_folder = './RK_forcing/'
RK_forcing_file = 'residence_times_flex.pkl'
RK_forcing_path = os.path.join(RK_forcing_folder, RK_forcing_file)
RK_forcing_residence_times_dict = joblib.load(RK_forcing_path)

D_values = [D for D in RK_forcing_residence_times_dict.keys() if D != 'ts']
print(D_values)

#max_res = max([max(RK_forcing_residence_times_dict[D]) for D in D_values if len(RK_forcing_residence_times_dict[D]) > 0])
max_res = 200
min_res = min([min(RK_forcing_residence_times_dict[D]) for D in D_values if len(RK_forcing_residence_times_dict[D]) > 0])
bins = np.linspace(min_res, max_res, 100)

D_values = [0.07444444444444444, 0.10666666666666666, 0.1388888888888889, 0.1711111111111111] 

fig, axes = plt.subplots(2, 2, figsize=(13, 6))

for i, D_value in enumerate(D_values):
    row = i // 2  # Riga dell'i-esimo subplot
    col = i % 2   # Colonna dell'i-esimo subplot
    
    RK_forcing_residence_times = RK_forcing_residence_times_dict.get(D_value, [])  # Ottiene la lista dei tempi di residenza per D_value
    if len(RK_forcing_residence_times) == 0:
        continue
    
    # Plot dell'istogramma nell'i-esimo subplot
    ax = axes[row, col]
    ax.hist(RK_forcing_residence_times, bins=bins, alpha=0.5, label=f'D = {round(D_value, 2)}', density=True)
    
    # Linea verticale su forcing period
    ax.axvline(x=forcing_period, color='black', linestyle='--', linewidth=1)
    
    # Aggiunta di etichette e legenda
    ax.set_ylabel('Density')
    ax.legend()

# Etichetta sull'asse x per l'ultimo subplot della seconda riga
axes[1, 0].set_xlabel('Residence times')
axes[1, 1].set_xlabel('Residence times')

# Aggiustamento dello spaziamento tra i subplot
plt.suptitle('Residence times distribution for increasing D values (flex threshold)')
plt.tight_layout()

# Mostra la figura
plt.savefig('immagini/residence_time_increasing_D_flex.png')

#Faccio lo stesso ma con threshold sul minimo

RK_residence_times_dict = joblib.load(RK_path_min)
Euler_residence_times_dict = joblib.load(Euler_path_min)

D_value = 0.1388888888888889

Euler_residence_times = Euler_residence_times_dict[D_value]
RK_residence_times = RK_residence_times_dict[D_value]

min_potential = fn.positive_min_quartic_potential([a_Euler, b_Euler])
second_der_min = fn.quartic_potential_2derivative(min_potential, [a_Euler, b_Euler])
second_der_max = fn.quartic_potential_2derivative(0, [a_Euler, b_Euler])
barrier_height = fn.potential(0, a_Euler, b_Euler) - fn.potential(min_potential, a_Euler, b_Euler)

bin_min = min(min(Euler_residence_times), min(RK_residence_times))
bin_max = max(max(Euler_residence_times), max(RK_residence_times))
#bin_max = 50
bins = np.linspace(bin_min, bin_max, 200)

kr_teorico = fn.kramer_rate(second_der_min, second_der_max, barrier_height, D_value)
theoretical_residence_times_distribution = [theoretical_residence_time(kr_teorico, bin) for bin in bins]

fig = plt.figure()
plt.hist(Euler_residence_times, bins=bins, alpha=0.5, label='Euler', density=True)
plt.hist(RK_residence_times, bins=bins, alpha=0.5, label='RK', density =True)
plt.plot(bins, theoretical_residence_times_distribution, label='Teorico')
plt.legend()
plt.savefig('immagini/residence_time_dist_min.png')

#Plotto gli istogrammi dei tempi di residenza per diversi valori del rumore e forzante periodica
#Legge il file di residence times di RK_forcing

RK_forcing_folder = './RK_forcing/'
RK_forcing_file = 'residence_times_min.pkl'
RK_forcing_path = os.path.join(RK_forcing_folder, RK_forcing_file)
RK_forcing_residence_times_dict = joblib.load(RK_forcing_path)

D_values = [D for D in RK_forcing_residence_times_dict.keys() if D != 'ts']
print(D_values)

#max_res = max([max(RK_forcing_residence_times_dict[D]) for D in D_values if len(RK_forcing_residence_times_dict[D]) > 0])
max_res = 200
min_res = min([min(RK_forcing_residence_times_dict[D]) for D in D_values if len(RK_forcing_residence_times_dict[D]) > 0])
bins = np.linspace(min_res, max_res, 100)


D_values = [0.07444444444444444, 0.10666666666666666, 0.1388888888888889, 0.1711111111111111] 

fig, axes = plt.subplots(2, 2, figsize=(13, 6))

for i, D_value in enumerate(D_values):
    row = i // 2  # Riga dell'i-esimo subplot
    col = i % 2   # Colonna dell'i-esimo subplot
    
    RK_forcing_residence_times = RK_forcing_residence_times_dict.get(D_value, [])  # Ottiene la lista dei tempi di residenza per D_value
    if len(RK_forcing_residence_times) == 0:
        continue
    
    # Plot dell'istogramma nell'i-esimo subplot
    ax = axes[row, col]
    ax.hist(RK_forcing_residence_times, bins=bins, alpha=0.5, label=f'D = {round(D_value, 2)}', density=True)
    
    # Linea verticale su forcing period
    ax.axvline(x=forcing_period, color='black', linestyle='--', linewidth=1)
    
    # Aggiunta di etichette e legenda
    ax.set_ylabel('Density')
    ax.legend()

# Etichetta sull'asse x per l'ultimo subplot della seconda riga
axes[1, 0].set_xlabel('Residence times')
axes[1, 1].set_xlabel('Residence times')

# Aggiustamento dello spaziamento tra i subplot
plt.suptitle('Residence times distribution for increasing D values (min threshold)')
plt.tight_layout()

# Mostra la figura
plt.savefig('immagini/residence_time_increasing_D_min.png')
plt.show()
"""
