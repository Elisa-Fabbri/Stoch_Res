import numpy as np
import sys
import os
import joblib
import matplotlib.pyplot as plt
import configparser


#Legge da terminale il file da leggere

autocorrelation_path = sys.argv[1]

directory = os.path.dirname(autocorrelation_path)

config = configparser.ConfigParser()
config_file = os.path.join(directory, 'configuration.txt')

config.read(config_file)

omega = float(config['simulation_parameters']['omega'])
h = float(config['simulation_parameters']['h'])

forcing_period = 2 * np.pi / omega

output_file = f'autocorrelation.pkl'
output_file = os.path.join(directory, output_file)

autocorrelation_data = joblib.load(autocorrelation_path)

ts = autocorrelation_data['ts']

for key in autocorrelation_data.keys():
    if key == 'ts':
        continue
    print(key)
    autocorrelation = np.array(autocorrelation_data[key])
    
    plt.plot(ts, autocorrelation, label=key)
    plt.xlabel('Time')
    plt.ylabel('Autocorrelation')
    plt.axvline(forcing_period, color='red', linestyle='--', label='Forcing period')
    plt.show()

    autocorrelation_fft = np.fft.fft(autocorrelation)
    freqs = np.fft.fftfreq(len(autocorrelation), d=h)

    positive_freqs = freqs[freqs > 0]
    positive_autocorrelation_fft = autocorrelation_fft[freqs > 0]

    # Calcolo della Power Spectral Density (PSD)
    power_spectrum = np.abs(positive_autocorrelation_fft)**2 / (len(autocorrelation) * h)

    expected_freq = omega/(2*np.pi)

    # Plot della Power Spectral Density (PSD)
    plt.plot(positive_freqs, power_spectrum, label='Power Spectral Density')
    plt.axvline(expected_freq, color='red', linestyle='--', label='Expected frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.show()


