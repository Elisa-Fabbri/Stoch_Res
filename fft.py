import numpy as np
import sys
import os
import joblib
import matplotlib.pyplot as plt
import configparser
import functions as fn


#Legge da terminale il file da leggere

autocorrelation_path = sys.argv[1]

directory = os.path.dirname(autocorrelation_path)

config = configparser.ConfigParser()
config_file = os.path.join(directory, 'configuration.txt')

config.read(config_file)

omega = float(config['simulation_parameters']['omega'])
h = float(config['simulation_parameters']['h'])

forcing_period = 2 * np.pi / omega
amplitude = float(config['simulation_parameters']['amplitude'])

a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])

output_file = f'autocorrelation.pkl'
output_file = os.path.join(directory, output_file)

autocorrelation_data = joblib.load(autocorrelation_path)

expected_freq = 1 / forcing_period

min = fn.positive_min_quartic_potential([a, b])
potential_second_derivative_min = fn.quartic_potential_2derivative(min, [a, b])
potential_second_derivative_max = fn.quartic_potential_2derivative(0, [a, b])

ts = autocorrelation_data['ts']

SNR_list = []

for key in autocorrelation_data.keys():
    if key == 'ts':
        continue
    print(key)
    autocorrelation = np.array(autocorrelation_data[key])

    # Plot dell'autocorrelazione
    plt.plot(ts, autocorrelation, label=key)
    plt.xlabel('Time')
    plt.ylabel('Autocorrelation')
    plt.axvline(forcing_period, color='red', linestyle='--', label='Forcing period')
    plt.legend()
    plt.show()

    # Trasformata di Fourier dell'autocorrelazione
    autocorrelation_fft = np.fft.fft(autocorrelation)
    n = autocorrelation.size

    #Trova passo temporale da ts
    h = ts[1] - ts[0]

    freqs = np.fft.fftfreq(n, d=h)

    # Filtriamo le frequenze positive
    positive_freqs = freqs[freqs > 0]
    positive_autocorrelation_fft = autocorrelation_fft[freqs > 0]

    # Calcolo della Power Spectral Density (PSD)
    power_spectrum = np.abs(positive_autocorrelation_fft)**2 / (len(autocorrelation) * h)

    max_index = np.argmin(np.abs(positive_freqs - expected_freq)) #indice più vicino alla frequenca attesa
    peak_value = power_spectrum[max_index]
    #Trovo la base del picco 
    num_neighbors = 2
    neighbors_indices = np.arange(max_index - num_neighbors, 
                                 max_index + num_neighbors + 1)
    
    peak_base = np.mean(power_spectrum[neighbors_indices])

    SNR = 2*peak_value / peak_base
    SNR_log = 10*np.log10(SNR)

    print(f'SNR: {SNR}')
    SNR_list.append((key, SNR, SNR_log))

    # Plot della Power Spectral Density (PSD)
    plt.plot(positive_freqs, power_spectrum, label='Power Spectral Density')
    plt.axvline(expected_freq, color='red', linestyle='--', label='Expected frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.show()

#Plotta il SNR 

SNR_theoretical_list = []
def theo_SNR(A_0, x_m, D, r_k):
    return np.pi*(A_0*x_m/D)**2 * r_k


for D in SNR_list:
    noise = D[0]
    rk = fn.kramer_rate(potential_second_derivative_min= potential_second_derivative_min,
                        potential_second_derivative_max= potential_second_derivative_max,
                        barrier_height= fn.potential(0, a, b) - fn.potential(min, a, b),
                        D=noise)
    SNR_theoretical = theo_SNR(amplitude, 1, noise, rk)
    SNR_theoretical_list.append(SNR_theoretical)

print(SNR_theoretical_list)
print(SNR_list)


plt.plot([x[0] for x in SNR_list], [x[1] for x in SNR_list], 'o')
plt.plot([x[0] for x in SNR_list], SNR_theoretical_list, linestyle='--')
plt.show()

plt.plot([x[0] for x in SNR_list], [x[2] for x in SNR_list], 'o')
plt.show()

#Plotta anche il SNR teorico, ovvero pi*(A_0*x_m/D)^2 * r_k



