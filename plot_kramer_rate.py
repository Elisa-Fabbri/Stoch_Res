"""
Questo modulo crea due grafici con il tasso di Kramer teorico e simulato con i due metodi di integrazione. 
Ciascun grafico corrisponde a un metodo diverso per binarizzare le traiettorie: flex e max.

Ho trovato che la threshold va messa sul flesso del potenziale e non sul massimo.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import pandas as pd
import functions as fn
import configparser

#Genero un grafico che confronta i tassi di fuga di Kramer ottenuti con i due metodi di integrazione e 
#threshold sul flesso

Euler_kramer_rate_df = pd.DataFrame(columns=['D', 'kramer_rate', 'std_kramer_rate'])
RK_kramer_rate_df = pd.DataFrame(columns=['D', 'kramer_rate', 'std_kramer_rate'])
for method in ['Euler', 'RK']:
    folder = f'./{method}_no_forcing/'
    file = 'residence_times_flex.pkl'
    path = os.path.join(folder, file)
    print(method)
    residence_times_dict = joblib.load(path)
    for key, residence_times in residence_times_dict.items():
        if key == 'ts':
            continue
        mean_residence_time = np.mean(residence_times) if len(residence_times) > 0 else 0
        std_residence_time = np.std(residence_times) if len(residence_times) > 0 else 0
        escape_rates = 1 / mean_residence_time if mean_residence_time != 0 else 0
        std_rates = std_residence_time / mean_residence_time**2 if mean_residence_time != 0 else 0

        df_to_concat = pd.DataFrame({'D': [key], 'kramer_rate': [escape_rates], 'std_kramer_rate': [std_rates]})
        if method == 'Euler':
            Euler_kramer_rate_df = pd.concat([Euler_kramer_rate_df, df_to_concat])
        else:
            RK_kramer_rate_df = pd.concat([RK_kramer_rate_df, df_to_concat])
Euler_kramer_rate_df = Euler_kramer_rate_df.sort_values(by='D')
RK_kramer_rate_df = RK_kramer_rate_df.sort_values(by='D')

# Controllo che i parametri siano uguali per i due metodi

config_file_Euler = './Euler_no_forcing/configuration.txt'
config_file_RK = './RK_no_forcing/configuration.txt'

config_Euler = configparser.ConfigParser()
config_RK = configparser.ConfigParser()

config_Euler.read(config_file_Euler)
config_RK.read(config_file_RK)

a_Euler = float(config_Euler['potential_parameters']['a'])
b_Euler = float(config_Euler['potential_parameters']['b'])
D_start_Euler = float(config_Euler['simulation_parameters']['D_start'])
D_end_Euler = float(config_Euler['simulation_parameters']['D_end'])
num_Ds_Euler = int(config_Euler['simulation_parameters']['num_Ds'])
a_RK = float(config_RK['potential_parameters']['a'])
b_RK = float(config_RK['potential_parameters']['b'])
D_start_RK = float(config_RK['simulation_parameters']['D_start'])
D_end_RK = float(config_RK['simulation_parameters']['D_end'])
num_Ds_RK = int(config_RK['simulation_parameters']['num_Ds'])

assert a_Euler == a_RK
assert b_Euler == b_RK
assert D_start_Euler == D_start_RK
assert D_end_Euler == D_end_RK
assert num_Ds_Euler == num_Ds_RK

# Calcolo il tasso di fuga di Kramer teorico

positive_min_quartic_potential = fn.positive_min_quartic_potential([a_Euler, b_Euler])

second_der_min = fn.quartic_potential_2derivative(positive_min_quartic_potential, [a_Euler, b_Euler])
second_der_max = fn.quartic_potential_2derivative(0, [a_Euler, b_Euler])
barrier_height = fn.potential(0, a_Euler, b_Euler) - fn.potential(positive_min_quartic_potential, a_Euler, b_Euler)

D = np.linspace(D_start_Euler, D_end_Euler, num_Ds_Euler)

kramer_rate = fn.kramer_rate(second_der_min, second_der_max, barrier_height, D)


plt.errorbar(Euler_kramer_rate_df['D'], Euler_kramer_rate_df['kramer_rate'], yerr=Euler_kramer_rate_df['std_kramer_rate'], 
             marker = 'o', linestyle = 'none', label = 'Euler')
plt.errorbar(RK_kramer_rate_df['D'], RK_kramer_rate_df['kramer_rate'], yerr=RK_kramer_rate_df['std_kramer_rate'], 
             marker = 'o', linestyle = 'none', label='RK')
plt.plot(D, kramer_rate, label='Kramer rate')
plt.legend()
plt.savefig('immagini/kramer_rate_flex_comparison.png')

#Faccio lo stesso ma per threshold sul massimo 

Euler_kramer_rate_df = pd.DataFrame(columns=['D', 'kramer_rate', 'std_kramer_rate'])
RK_kramer_rate_df = pd.DataFrame(columns=['D', 'kramer_rate', 'std_kramer_rate'])

for method in ['Euler', 'RK']:
    folder = f'./{method}_no_forcing/'
    file = 'residence_times_max.pkl'
    path = os.path.join(folder, file)
    print(method)
    residence_times_dict = joblib.load(path)
    for key, residence_times in residence_times_dict.items():
        if key == 'ts':
            continue
        mean_residence_time = np.mean(residence_times) if len(residence_times) > 0 else 0
        std_residence_time = np.std(residence_times) if len(residence_times) > 0 else 0
        escape_rates = 1 / mean_residence_time if mean_residence_time != 0 else 0
        std_rates = std_residence_time / mean_residence_time**2 if mean_residence_time != 0 else 0

        df_to_concat = pd.DataFrame({'D': [key], 'kramer_rate': [escape_rates], 'std_kramer_rate': [std_rates]})
        if method == 'Euler':
            Euler_kramer_rate_df = pd.concat([Euler_kramer_rate_df, df_to_concat])
        else:
            RK_kramer_rate_df = pd.concat([RK_kramer_rate_df, df_to_concat])
Euler_kramer_rate_df = Euler_kramer_rate_df.sort_values(by='D')
RK_kramer_rate_df = RK_kramer_rate_df.sort_values(by='D')

# Plot dei risultati contro il tasso di fuga di Kramer teorico

plt.figure()
plt.errorbar(Euler_kramer_rate_df['D'], Euler_kramer_rate_df['kramer_rate'], yerr=Euler_kramer_rate_df['std_kramer_rate'], 
             marker = 'o', linestyle = 'none', label = 'Euler')
plt.errorbar(RK_kramer_rate_df['D'], RK_kramer_rate_df['kramer_rate'], yerr=RK_kramer_rate_df['std_kramer_rate'],
                marker = 'o', linestyle = 'none', label='RK')
plt.plot(D, kramer_rate, label='Kramer rate')
plt.legend()
plt.savefig('immagini/kramer_rate_max_comparison.png')
plt.show()

#Faccio lo stesso con threshold sul minimo

Euler_kramer_rate_df = pd.DataFrame(columns=['D', 'kramer_rate', 'std_kramer_rate'])
RK_kramer_rate_df = pd.DataFrame(columns=['D', 'kramer_rate', 'std_kramer_rate'])

for method in ['Euler', 'RK']:
    folder = f'./{method}_no_forcing/'
    file = 'residence_times_min.pkl'
    path = os.path.join(folder, file)
    print(method)
    residence_times_dict = joblib.load(path)
    for key, residence_times in residence_times_dict.items():
        if key == 'ts':
            continue
        mean_residence_time = np.mean(residence_times) if len(residence_times) > 0 else 0
        std_residence_time = np.std(residence_times) if len(residence_times) > 0 else 0
        escape_rates = 1 / mean_residence_time if mean_residence_time != 0 else 0
        std_rates = std_residence_time / mean_residence_time**2 if mean_residence_time != 0 else 0

        df_to_concat = pd.DataFrame({'D': [key], 'kramer_rate': [escape_rates], 'std_kramer_rate': [std_rates]})
        if method == 'Euler':
            Euler_kramer_rate_df = pd.concat([Euler_kramer_rate_df, df_to_concat])
        else:
            RK_kramer_rate_df = pd.concat([RK_kramer_rate_df, df_to_concat])
Euler_kramer_rate_df = Euler_kramer_rate_df.sort_values(by='D')
RK_kramer_rate_df = RK_kramer_rate_df.sort_values(by='D')

# Plot dei risultati contro il tasso di fuga di Kramer teorico

plt.figure()
plt.errorbar(Euler_kramer_rate_df['D'], Euler_kramer_rate_df['kramer_rate'], yerr=Euler_kramer_rate_df['std_kramer_rate'], 
             marker = 'o', linestyle = 'none', label = 'Euler')
plt.errorbar(RK_kramer_rate_df['D'], RK_kramer_rate_df['kramer_rate'], yerr=RK_kramer_rate_df['std_kramer_rate'],
                marker = 'o', linestyle = 'none', label='RK')
plt.plot(D, kramer_rate, label='Kramer rate')
plt.legend()
plt.savefig('immagini/kramer_rate_min_comparison.png')
plt.show()
    
