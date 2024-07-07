import numpy as np
import configparser
import functions as fn
import sys
import os
import aesthetics as aes
import joblib
import shutil

#Read the configuration file:-----------------------------------------------------------------------------------------

config = configparser.ConfigParser()

config_file = sys.argv[1]

if not os.path.isfile(config_file):
    with aes.red_text():
        if config_file == 'configuration.txt':
            print('Error: The default configuration file "configuration.txt" does not exist in the current folder!')
        else:
            print(f'Error: The specified configuration file "{config_file}" does not exist in the current folder!')
        sys.exit()

config.read(config_file)

#Quartic potential parameters
a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])

#Method to solve the differential equation
method = config['simulation_parameters']['method']

#Initial condition
x_0 = float(config['simulation_parameters']['x_0'])
t_0 = float(config['simulation_parameters']['t_0'])
split_x_0 = config['simulation_parameters'].getboolean('split_x_0') #If True, the initial condition is split in half
normal_x_0 = config['simulation_parameters'].getboolean('normal_x_0') #If True, the initial are normally distributed around x_0

#Component of the system
noise = config['simulation_parameters'].getboolean('noise') #If True, noise is added to the system
if noise == True:
    D_start = float(config['simulation_parameters']['D_start'])
    D_end = float(config['simulation_parameters']['D_end'])
    num_Ds = int(config['simulation_parameters']['num_Ds'])
    random_seed = int(config['simulation_parameters']['random_seed'])
    np.random.seed(random_seed)
else:
    D_start = 0
    D_end = 0
    num_Ds = 1


Ds = np.linspace(D_start, D_end, num_Ds)

periodic_forcing = config['simulation_parameters'].getboolean('periodic_forcing') #If True, a periodic signal is added to the system

if periodic_forcing == True:
    amplitude = float(config['simulation_parameters']['amplitude'])
    omega = float(config['simulation_parameters']['omega'])
else:
    amplitude = 0
    omega = 0

#Simulation parameters
t_end = float(config['simulation_parameters']['t_end'])
h = float(config['simulation_parameters']['h'])
if noise == True:
    num_simulations = int(config['simulation_parameters']['num_simulations'])
else:
    num_simulations = 1

#Output file

results_directory = config['simulation_directory']['output_directory']

#Clean the results directory:-----------------------------------------------------------------------------------------

if os.path.isdir(results_directory):
    shutil.rmtree(results_directory)

os.mkdir(results_directory)

#Copia il file di configurazione nella cartella dei risultati
shutil.copy(config_file, results_directory)

#-----------------------------------------------------

#Simulation:

results_dict = {}

for D in Ds:
    if method == 'RK4':
        ts, ys = fn.stochRK4(fn.system, t_end, h, x_0, D, [a, b], num_simulations, t_0, 
                             amplitude, omega, noise, 
                             normal_x_0,
                             split_x_0)
    elif method == 'euler':
        ts, ys = fn.euler(fn.system, t_end, h, x_0, D, [a, b], num_simulations, t_0, 
                          amplitude, omega, noise, 
                          normal_x_0, 
                          split_x_0)
    else:
        with aes.red_text():
            print('Error: The method specified in the configuration file is not valid!')
            sys.exit()
    
    results_dict[D] = ys
    results_dict['ts'] = ts

joblib.dump(results_dict, f'{results_directory}/trajectories.pkl')

