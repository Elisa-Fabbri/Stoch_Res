"""
This module contains the functions to simulate the stochastic resonance in a bistable potential 
and analyze the results.
"""

import numpy as np


def stochRK4(fn, t_end, h, y_0, D, parameters, N, 
             t_0=0., A=0, omega=0, noise_bool=True, 
             normal_x_0=True, split_x_0=False):
    """
    Implementazione del metodo di Runge-Kutta di ordine 4 per equazioni differenziali stocastiche (SDE).

    :param fn: Funzione che definisce il sistema di equazioni differenziali.
    :param t_end: Tempo finale della simulazione.
    :param h: Passo temporale dell'integratore.
    :param y_0: Condizione iniziale.
    :param D: Metà della varianza del rumore.
    :param parameters: Parametri aggiuntivi della funzione fn.
    :param N: Numero di simulazioni indipendenti.
    :param t_0: Tempo iniziale (default 0).
    :param A: Ampiezza del segnale periodico (default 0).
    :param omega: Frequenza del segnale periodico (default 0).
    :param noise_bool: Booleano per l'abilitazione del rumore (default True).
    :param normal_x_0: Booleano per l'uso di rumore normale nella condizione iniziale (default True).
    :param split_x_0: Booleano per la suddivisione delle condizioni iniziali (default False).
    :return: Una coppia di array (ts, ys) con i tempi di simulazione e le soluzioni simulate.
    """

    ts = np.arange(t_0, t_end, h)  # Array di tempi della simulazione
    ys = np.zeros((N, len(ts)))  # Array di soluzioni

    # Imposta le condizioni iniziali
    if not split_x_0:
        y_0 = np.array([y_0] * N)
    else:
        y_0 = np.array([y_0] * (N // 2) + [-y_0] * (N // 2))

    ys[:, 0] = y_0

    # Genera una fase iniziale casuale per ogni simulazione
    initial_phases = np.random.uniform(0, 2 * np.pi, N)

    # Simulazione del processo
    for i in range(len(ts) - 1):  # L'ultima iterazione non ha successore
        t = ts[i]
        
        # Calcola i termini di Runge-Kutta
        k1 = fn(t, ys[:, i], parameters, A, omega, initial_phases)
        k2 = fn(t + h / 2., ys[:, i] + h * k1 / 2., parameters, A, omega, initial_phases)
        k3 = fn(t + h / 2., ys[:, i] + h * k2 / 2., parameters, A, omega, initial_phases)
        k4 = fn(t + h, ys[:, i] + h * k3, parameters, A, omega, initial_phases)
        
        # Update della soluzione
        ys[:, i + 1] = ys[:, i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.

        # Aggiunge rumore stocastico se abilitato
        if noise_bool:
            noise = np.random.normal(size=N)
            ys[:, i + 1] += np.sqrt(2 * D * h) * noise
    
    return ts, ys


def euler(fn, t_end, h, y_0, D, parameters, N, t_0=0., A=0, omega=0, noise_bool=True, normal_x_0=True, split_x_0=False):
    
    ts = np.arange(t_0, t_end, h)
    ys = np.zeros((N, len(ts)))

    if split_x_0 == False:
        y_0 = np.array([y_0] * N)
    else:
        y_0 = np.array([y_0] * (N // 2) + ([- y_0]) * (N // 2))
    
    ys[:, 0] = y_0
    
    if noise_bool == True and normal_x_0 == True:
        stoch_step = True
    else:
        stoch_step = False
    
    # Genera una fase iniziale casuale per ogni simulazione
    initial_phases = np.random.uniform(0, 2 * np.pi, N)

    for i, t in enumerate(ts[:-1]):  # Use ts[:-1] to avoid IndexError on ys[i+1]
        if stoch_step == True:
            noise = np.random.normal(size=N)
            ys[:, i] += np.sqrt(2 * D * h) * noise
        if noise_bool == True and stoch_step == False:
            stoch_step = True
        
        ys[:, i+1] = ys[:, i] + fn(t, ys[:, i], parameters, A, omega, initial_phases) * h
    
    return ts, ys



def potential(x, a, b):
    return - (1/2)*a*x**2 + (1/4)*b*x**4

def periodic_signal(t, amplitude, omega):
    return amplitude * np.cos(omega * t)

# Definizione della derivata del potenziale quartico
def quartic_potential_derivative(x, a, b):
    return b*x**3 - a*x

def quartic_potential_2derivative(X, parameters):
    a, b = parameters
    return -a + 3*b*X**2

def positive_min_quartic_potential(parameters):
    a, b = parameters
    return np.sqrt(a/(b))

def positive_flex_quartic_potential(parameters):
    a, b = parameters
    return np.sqrt(a/(3*b))

def system(t, x, parameters, amplitude=0, omega=0, phases=None):
    a, b = parameters
    potential_derivative = -quartic_potential_derivative(x, a, b)
    
    if phases is None:
        combined_term = potential_derivative + amplitude * np.cos(omega * t)
    else:
        combined_term = potential_derivative + amplitude * np.cos(omega * t + phases)
    
    return np.array([combined_term])

def binarize_trajectory(trajectory, positive_threshold, negative_threshold):

    all_binarized_traj = np.zeros_like(trajectory)
    for num_sim in range(0, trajectory.shape[0]):

        traj_i = trajectory[num_sim, :]
        #print('Trajectory is:', traj_i)
        x_0 = traj_i[0]
        #print('x 0 is:', x_0)

        mean_threshold = (positive_threshold + negative_threshold) / 2
        if x_0 > mean_threshold:
            current_state = 1
        else:
            current_state = -1


        binarized_traj_i= [current_state]

        for x in traj_i[1:] : #traj[1:]:
            if x < negative_threshold and current_state == 1:
                current_state = -1
            elif x > positive_threshold and current_state == -1:
                current_state = 1
            binarized_traj_i.append(current_state)
        #print('Num simulation', num_sim)
        #print('Binarized trajectory is', binarized_traj_i[:10])
        all_binarized_traj[num_sim, :] = binarized_traj_i
    return all_binarized_traj

def find_residence_time(binarized_trajectory, ts):

    residence_times = []

    for num_sim in range(0, binarized_trajectory.shape[0]):
        traj = binarized_trajectory[num_sim, :]
        current_state = traj[0]
        crossing_time = ts[0]
        for i, state in enumerate(traj):
            if state != current_state:
                residence_times.append(ts[i] - crossing_time)
                crossing_time = ts[i]
                current_state = state
    return residence_times

def kramer_rate(potential_second_derivative_min, 
                potential_second_derivative_max, 
                barrier_height,
                D):
    
    prefactor = (potential_second_derivative_min * (- potential_second_derivative_max)) / (2 * np.pi)
    return prefactor*np.exp(- barrier_height / D)

def kramer_rate_simplified(barrier_height,
                           D):
    """
    This should be valid for a symmetric - bistable potential with 
    a = 1 and b = 1
    """
    prefactor = np.sqrt(2) * np.pi
    return prefactor*np.exp(- barrier_height / D)

def find_D_from_kramer_rate(barrier_height,
                            potential_second_derivative_min,
                            potential_second_derivative_max,
                            kramer_rate):
    
    log_argument = (2 * np.pi * kramer_rate) / (potential_second_derivative_min * (- potential_second_derivative_max))
    log = np.log(log_argument)
    return - (barrier_height / log)
    
def calculate_escape_rates(residence_times):
    escape_rates = np.zeros_like(residence_times[:, 0], dtype=float)  # Array per memorizzare i tassi di fuga
    
    # Calcolare il reciproco solo se il denominatore non è zero
    non_zero_mask = residence_times[:, 0] != 0
    escape_rates[non_zero_mask] = 1 / residence_times[non_zero_mask, 0]
    
    return escape_rates

def calculate_escape_rates_err(residence_times):
    escape_rates_err = np.zeros_like(residence_times[:, 0], dtype=float)  # Array per memorizzare gli errori sui tassi di fuga
    
    # Calcolare l'errore solo se il denominatore non è zero
    non_zero_mask = residence_times[:, 0] != 0
    escape_rates_err[non_zero_mask] = residence_times[non_zero_mask, 1] / residence_times[non_zero_mask, 0]**2
    
    return escape_rates_err

"""
def stochRK4(fn,  # Funzione che definisce il sistema di equazioni differenziali
             t_end,  # Tempo finale della simulazione
             h,  # Passo temporale dell'integratore
             y_0,  # Condizione iniziale
             D,  # Metà varianza del rumore
             parameters,  # Parametri aggiuntivi della funzione
             N,  # numero di simulazioni 
             t_0=0.,
             A=0,
             omega=0,
             noise_bool=True,
             normal_x_0=True,
             split_x_0=False):

    lh = h / 2.  # Define leapfrog "half step"

    ts = np.arange(t_0, t_end, lh)  # Array di tempi della simulazione
    ys = np.zeros((N, len(ts)))  # Array di soluzioni

    if split_x_0 == False:
        y_0 = np.array([y_0] * N)
    else:
        y_0 = np.array([y_0] * (N // 2) + ([- y_0]) * (N // 2))

    ys[:, 0] = y_0

    if noise_bool and normal_x_0:
        stoch_step = True
    else:
        stoch_step = False

    # Genera una fase iniziale casuale per ogni simulazione
    initial_phases = np.random.uniform(0, 2 * np.pi, N)

    for i, t in enumerate(ts):  # Per ogni tempo della simulazione
        if stoch_step:
            noise = np.random.normal(size=N)
            ys[:, i] += np.sqrt(2 * D * h) * noise  # Aggiunge rumore a passi alternati
        
        # Calcola k1, k2, k3, k4 con le fasi iniziali differenti
        k1 = fn(t, ys[:, i], parameters, A, omega, initial_phases)
        k2 = fn(t + lh / 2., ys[:, i] + lh * k1 / 2., parameters, A, omega, initial_phases)
        k3 = fn(t + lh / 2., ys[:, i] + lh * k2 / 2., parameters, A, omega, initial_phases)
        k4 = fn(t + lh, ys[:, i] + lh * k3, parameters, A, omega, initial_phases)

        try:
            ys[:, i + 1] = ys[:, i] + lh * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        except IndexError:
            return ts, ys
        
        if noise_bool:
            stoch_step = not stoch_step

    return ts, ys
"""