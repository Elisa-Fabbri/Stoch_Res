"""
This module is used for plotting the trajectories of the signal
"""

import numpy as np
import pandas as pd
import sys
import joblib
import matplotlib.pyplot as plt

traj_path = 'RK_forcing/trajectories.pkl'

#Load trajectories

traj_dict = joblib.load(traj_path)

ts = traj_dict['ts']

#Elimino ts dal dizionario
traj_dict.pop('ts')

print('Number of trajectories:', len(traj_dict.keys()))

#Faccio un grafico con 8 subplot per contenere le prime 8 traiettorie

fig, axs = plt.subplots(4, 2, figsize=(15, 15))
axs = axs.ravel()

for i, key in enumerate(traj_dict.keys()):
    if i >= 8:
        break
    value = traj_dict[key]
    traj = value[0]
    axs[i].plot(ts, traj)
    axs[i].set_title('Noise intensity: ' + str(round(float(key), 3)))
plt.tight_layout()
plt.show()
plt.close()

#Faccio un grafico con 8 subplot per contenere le ultime 8 traiettorie

fig, axs = plt.subplots(4, 2, figsize=(15, 15))
axs = axs.ravel()

for i, key in enumerate(traj_dict.keys()):
    if i < 8:
        continue
    value = traj_dict[key]
    traj = value[0]
    axs[i - 8].plot(ts, traj)
    axs[i - 8].set_title('Noise intensity: ' + str(round(float(key), 3)))
plt.tight_layout()
plt.show()
plt.close()


