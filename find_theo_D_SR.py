import numpy as np
import functions as fn
from scipy.optimize import root
import configparser
import sys

config_file = sys.argv[1]

config = configparser.ConfigParser()
config.read(config_file)

a = float(config['potential_parameters']['a'])
b = float(config['potential_parameters']['b'])
omega = float(config['simulation_parameters']['omega'])
parameters = [a, b]


def transcendental_equation(D_SR, omega, delta_V, potential_second_derivative_min, potential_second_derivative_max, barrier_height):
    r_k = fn.kramer_rate(potential_second_derivative_min, potential_second_derivative_max, barrier_height, D_SR)
    lhs = 4 * r_k**2
    rhs = omega**2 * ((delta_V / D_SR) - 1)
    return lhs - rhs


D_SR_initial_guess = 0.05

potential_second_derivative_min = fn.quartic_potential_2derivative(fn.positive_min_quartic_potential(parameters), parameters)
potential_second_derivative_max = fn.quartic_potential_2derivative(0, parameters)
barrier_height = - fn.potential(fn.positive_min_quartic_potential(parameters), *parameters) + fn.potential(0, *parameters)
print('Barrier height is:', barrier_height)

# Risoluzione dell'equazione trascendentale
solution = root(transcendental_equation, D_SR_initial_guess, args=(omega, 
                                                                   barrier_height, 
                                                                   potential_second_derivative_min, 
                                                                   potential_second_derivative_max, 
                                                                   barrier_height))

if solution.success:
    print("Soluzione trovata:", solution.x[0])
else:
    print("La soluzione non è stata trovata:", solution.message)

print('If the prefactor of the Kramers rate is independent of D, the value that maximizes the SNR is:',
       barrier_height/2 )