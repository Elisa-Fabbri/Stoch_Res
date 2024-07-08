"""
Questo modulo legge i risultati della simulazione, salvati in un dizionario. 
Il dizionario in input contiene le chiavi che indicano il valore del parametro D e come valori corrispondenti 
dei numpy array che contengono le traiettorie simulate (ogni riga corrisponde a una diversa simulazione).
L'unico valore che non è una traiettoria è 'ts' che indica il tempo.

Il modulo restituisce un dizionario con le stesse chiavi, ma i valori sono dei numpy array binarizzati secondo
il metodo indicato: se max la soglia viene messa sul massimo del potenziale (0), se flex la soglia viene messa
sui valori di flesso del potenziale.

Il dizionario restituito contiene le stesse chiavi del dizionario in input, ma i valori sono dei numpy array binarizzati
secondo il metodo scelto.
Il dizionario restituito contiene anche la chiave 'ts' lasciata invariata.

Il file in cui viene salvato il dizionario è lo stesso del file in input, ma con '_binarized' e il metodo di binarizzazione
aggiunto al nome del file.

Il modulo va eseguito come:

python binarize_trajectories.py trajectory_file threshold_choice 

dove:
- trajectory_file è il file dove è salvato il dizionario da binarizzare
- threshold_choice è la scelta del metodo di binarizzazione: 'max' o 'flex'

"""


import os
import joblib
import sys
import configparser
import functions as fn
import aesthetics as aes

#Legge da terminale il file da leggere

trajectory_file = sys.argv[1]
threshold_choice = sys.argv[2] #max or flex or min

#Leggi il nome della directory da trajectory_file
directory = os.path.dirname(trajectory_file)
config_file = os.path.join(directory, 'configuration.txt')

#Legge i parametri del potenziale dal file di configurazione

config = configparser.ConfigParser()
config.read(config_file)

a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])
parameters = [a, b]

positive_flex = fn.positive_flex_quartic_potential(parameters)
pos_min = fn.positive_min_quartic_potential(parameters)

if threshold_choice == 'max':
    positive_threshold = negative_threshold = 0
elif threshold_choice == 'flex':
    positive_threshold = positive_flex
    negative_threshold = -positive_flex
elif threshold_choice == 'min':
    positive_threshold = pos_min
    negative_threshold = -pos_min
else:
    with aes.red_text():
        print('Error: The threshold choice must be either "max" or "flex"!')
        sys.exit()


#Output file nella stessa cartella di trajectory_file
output_directory = os.path.dirname(trajectory_file)
output_file = f'binarized_{threshold_choice}.pkl'
output_file = os.path.join(output_directory, output_file)

trajectories_to_binarize = joblib.load(trajectory_file)

#Ciascuna chiave del dizionario indica il valore del parametro D tranne la chiave 'ts' che indica il tempo

for key, trajectory in trajectories_to_binarize.items():
    if key == 'ts':
        continue
    #Per ciascun rumore ci sono num_simulations traiettorie
    trajectories_to_binarize[key] = fn.binarize_trajectory(trajectory, positive_threshold, negative_threshold)

joblib.dump(trajectories_to_binarize, output_file)
    






