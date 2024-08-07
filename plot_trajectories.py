"""
This module is used for plotting the trajectories of the signal
"""

import numpy as np
import pandas as pd
import sys
import joblib
import matplotlib.pyplot as plt
import os
import functions as fn
import sys
import configparser
import aesthetics as aes

#Legge con sys il path alle traiettorie

traj_path = sys.argv[1]
traj_folder = os.path.dirname(traj_path)

#Load trajectories

traj_dict = joblib.load(traj_path)

ts = traj_dict['ts']

#Elimino ts dal dizionario
traj_dict.pop('ts')

print('Number of trajectories:', len(traj_dict.keys()))

#Creo una cartella per contenere le immagini

image_folder = os.path.join(traj_folder, 'immagini_trajectories')


if not os.path.exists(image_folder):
    os.makedirs(image_folder)


number_of_trajectories = len(traj_dict.keys())

config_file = os.path.join(traj_folder, 'configuration.txt')

#Read the configuration file:----------------------------------------------------------------------------------------

config = configparser.ConfigParser()

config.read(config_file)

D_start = float(config['simulation_parameters']['D_start'])
D_end = float(config['simulation_parameters']['D_end'])
D_step = int(config['simulation_parameters']['num_Ds'])

print('I valori di rumore simulati sono:', np.linspace(D_start, D_end, D_step))

# Leggi gli indici dal terminale
input_indices = input("Enter the indices of noise intensity values you want to plot (separated by commas): ")

# Elabora l'input dell'utente
try:
    # Converte l'input in una lista di interi
    chosen_keys_indices = list(map(int, input_indices.split(',')))
    
    # Filtra solo gli indici validi
    chosen_keys_indices = [i for i in chosen_keys_indices if 0 <= i < number_of_trajectories]
    
    # Ordina gli indici in ordine crescente
    chosen_keys_indices = np.sort(chosen_keys_indices)

    # Estrai i valori corrispondenti agli indici scelti
    chosen_keys = [list(traj_dict.keys())[i] for i in chosen_keys_indices]
    
except ValueError:
    print("Please enter valid integers separated by commas.")
    chosen_keys = []

#Runno una simulazione per ciascun valore di noise intensity scelto con stessi parametri 
#delle simulazioni precedenti

t_end = float(config['simulation_parameters']['t_end'])//100
h = float(config['simulation_parameters']['h'])
x_0 = float(config['simulation_parameters']['x_0'])
t_0 = float(config['simulation_parameters']['t_0'])
a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])
num_simulations = 2
amplitude = float(config['simulation_parameters']['amplitude'])
omega = float(config['simulation_parameters']['omega'])
normal_x_0 = config['simulation_parameters'].getboolean('normal_x_0')
split_x_0 = config['simulation_parameters'].getboolean('split_x_0')

results_dict = {}

for key in chosen_keys:
    noise = float(key)
    ts, ys = fn.stochRK4(fn.system, t_end, h, x_0, float(key), [a, b], num_simulations, t_0,
                            amplitude, omega, noise,
                            normal_x_0,
                            split_x_0,
                            random_phase=False)
    results_dict[key] = ys
    results_dict['ts'] = ts

#Plotto le traiettorie (3 subplot)

fig, axs = plt.subplots(3, 1, figsize=(15, 15))
axs = axs.ravel()

for i, key in enumerate(chosen_keys):
    value = results_dict[key]
    traj = value[0]
    axs[i].plot(ts, traj, label='Noise intensity: ' + str(round(float(key), 3)), linewidth=0.5)
    #Plotto anche la forzante periodica
    axs[i].plot(ts, amplitude * np.cos(omega * ts), label='Periodic forcing', linewidth=1, alpha=0.5)
    #Plotto la forzante periodica normalizzata tra -1 e 1
    axs[i].plot(ts, np.cos(omega * ts), label='Normalized periodic forcing', linewidth=1, linestyle='--', alpha=0.5)
    axs[i].set_title('Noise intensity: ' + str(round(float(key), 3)), fontsize = 16)
plt.tight_layout()
#plt.show()
filename = 'traiettorie_scelte'
plt.savefig(os.path.join(image_folder, filename))
plt.close()

fig,axs = plt.subplots(3, 1, figsize=(15,15))
axs = axs.ravel()

for i, key in enumerate(chosen_keys):
    trajectories = results_dict[key]
    results_dict[key] = fn.binarize_trajectory(trajectories,1,-1)
    value = results_dict[key]
    traj = value[0]
    axs[i].plot(ts, traj, label='Noise intensity: ' + str(round(float(key), 3)), linewidth=1)
    #Plotto anche la forzante periodica
    axs[i].plot(ts, amplitude * np.cos(omega * ts), label='Periodic forcing', linewidth=1, alpha=0.5)
    #Plotto la forzante periodica normalizzata tra -1 e 1
    axs[i].plot(ts, np.cos(omega * ts), label='Normalized periodic forcing', linewidth=1, linestyle='--', alpha=0.5)
    axs[i].set_title('Noise intensity: ' + str(round(float(key), 3)), fontsize = 16)
plt.tight_layout()
#plt.show()
filename = 'traiettorie_scelte_bin'
plt.savefig(os.path.join(image_folder, filename))
plt.close()
