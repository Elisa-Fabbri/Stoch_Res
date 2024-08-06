import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import configparser

# Load data from files
residence_times_file_1 = './RK_forcing_0.4/residence_times_min.pkl'
residence_times_file_2 = './RK_forcing_0.2/residence_times_min.pkl'

# Read dictionaries from the files
residence_times_dict_1 = joblib.load(residence_times_file_1)
residence_times_dict_2 = joblib.load(residence_times_file_2)

# Get the directory paths
directory1 = os.path.dirname(residence_times_file_1)
directory2 = os.path.dirname(residence_times_file_2)

# Configuration file paths
configuration_file1 = os.path.join(directory1, 'configuration.txt')
configuration_file2 = os.path.join(directory2, 'configuration.txt')

# Read the configuration files
config1 = configparser.ConfigParser()
config1.read(configuration_file1)

config2 = configparser.ConfigParser()
config2.read(configuration_file2)

# Extract simulation parameters from configuration files
D_start1 = float(config1['simulation_parameters']['D_start'])
D_end1 = float(config1['simulation_parameters']['D_end'])
num_Ds1 = int(config1['simulation_parameters']['num_Ds'])

D_start2 = float(config2['simulation_parameters']['D_start'])
D_end2 = float(config2['simulation_parameters']['D_end'])
num_Ds2 = int(config2['simulation_parameters']['num_Ds'])

# Create D value arrays
D_values1 = np.linspace(D_start1, D_end1, num_Ds1)
D_values2 = np.linspace(D_start2, D_end2, num_Ds2)

# Forcing period calculation
omega = float(config1['simulation_parameters']['omega'])
forcing_period = (2 * np.pi) / omega

# Group D values into 3 groups of 6
D_values_group1 = D_values1[:6]
D_values_group2 = D_values1[6:12]
D_values_group3 = D_values1[12:18]
D_values_group4 = D_values1[18:]

D_values_group5 = D_values2[:6]
D_values_group6 = D_values2[6:12]
D_values_group7 = D_values2[12:18]
D_values_group8 = D_values2[18:]

def plot_histograms(D_values, title, filename, residence_times_dict, forcing_period):
    num_plots = len(D_values)
    num_rows = 3
    num_cols = 2
    num_subplots = num_rows * num_cols

    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(13, 10))
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    # Ensure we do not attempt to access more subplots than created
    for i, D_value in enumerate(D_values):
        if i >= num_subplots:
            print(f"Warning: Too many D values ({num_plots}) for {num_subplots} subplots.")
            break

        ax = axes[i]  # Access the subplot at index i

        # Get residence times for D_value
        residence_times = residence_times_dict.get(D_value, [])
        if len(residence_times) == 0:
            ax.set_visible(False)  # Hide subplot if no data
            continue

        # Normalize residence times
        normalized_residence_times = np.array(residence_times) / forcing_period

        # Plot histogram
        ax.hist(normalized_residence_times, bins=100, alpha=0.5, 
                label=f'D = {round(D_value, 3)}', range=(0, 5))

        # Add vertical line at T = forcing_period
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

        # Add labels and legend
        ax.set_ylabel('Counts')
        ax.set_xlabel(r'Residence times $T/T_{\text{forcing}}$')
        ax.legend(loc='upper right')

    # Hide any unused subplots
    for j in range(i + 1, num_subplots):
        axes[j].set_visible(False)

    # Set title and adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leavespace for suptitle

    # Save and close figure
    plt.savefig(filename)
    plt.close()

# Image directories
image_directory1 = os.path.join(directory1, 'images')
image_directory2 = os.path.join(directory2, 'images')

# Create directories if they don't exist
os.makedirs(image_directory1, exist_ok=True)
os.makedirs(image_directory2, exist_ok=True)

# Plot histograms for different groups of D values
plot_histograms(D_values_group1, 
                'Residence times distribution for first 6 D values (normalized by forcing period)', 
                os.path.join(image_directory1, 'res_times_group1.png'), 
                residence_times_dict_1, forcing_period)

plot_histograms(D_values_group2, 
                'Residence times distribution for next 6 D values (normalized by forcing period)', 
                os.path.join(image_directory1, 'res_times_group2.png'), 
                residence_times_dict_1, forcing_period)

plot_histograms(D_values_group3, 
                'Residence times distribution for last 6 D values (normalized by forcing period)', 
                os.path.join(image_directory1, 'res_times_group3.png'), 
                residence_times_dict_1, forcing_period)

plot_histograms(D_values_group4, 
                'Residence times distribution for first 6 D values (normalized by forcing period)', 
                os.path.join(image_directory1, 'res_times_group4.png'), 
                residence_times_dict_1, forcing_period)

plot_histograms(D_values_group5, 
                'Residence times distribution for next 6 D values (normalized by forcing period)', 
                os.path.join(image_directory2, 'res_times_group1.png'), 
                residence_times_dict_2, forcing_period)

plot_histograms(D_values_group6, 
                'Residence times distribution for last 6 D values (normalized by forcing period)', 
                os.path.join(image_directory2, 'res_times_group2.png'), 
                residence_times_dict_2, forcing_period)

plot_histograms(D_values_group7,
               	'Residence times distribution for last 6 D values (normalized by forcing period)',
                os.path.join(image_directory2, 'res_times_group3.png'),
		residence_times_dict_2, forcing_period)

plot_histograms(D_values_group8,
               'Residence times distribution for last 6 D values (normalized by forcing period)',
                os.path.join(image_directory2, 'res_times_group4.png'),
                residence_times_dict_2, forcing_period)

#For each residence times dictionary do a plot with 4 subplots, consisting in the 3rd element of each group of 6 D values

# Plot histograms for different groups of D values

group1 = [D_values_group1[2], D_values_group2[2], D_values_group3[2], D_values_group4[2]]
group2 = [D_values_group5[2], D_values_group6[2], D_values_group7[2], D_values_group8[2]]

def plot_hist_for_groups(D_values, title, filename, residence_times_dict, forcing_period):
    num_plots = len(D_values)
    num_rows = 2
    num_cols = 2
    num_subplots = num_rows * num_cols

    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(13, 10))
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    # Ensure we do not attempt to access more subplots than created
    for i, D_value in enumerate(D_values):
        if i >= num_subplots:
            print(f"Warning: Too many D values ({num_plots}) for {num_subplots} subplots.")
            break

        ax = axes[i]  # Access the subplot at index i

        # Get residence times for D_value
        residence_times = residence_times_dict.get(D_value, [])
        if len(residence_times) == 0:
            ax.set_visible(False)  # Hide subplot if no data
            continue

        # Normalize residence times
        normalized_residence_times = np.array(residence_times) / forcing_period

        # Plot histogram
        ax.hist(normalized_residence_times, bins=100, alpha=0.5, 
                label=f'D = {round(D_value, 3)}', range=(0, 5))

        # Add vertical line at T = forcing_period
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

        # Add labels and legend
        ax.set_ylabel('Counts')
        ax.set_xlabel(r'Residence times $T/T_{\text{forcing}}$')
        ax.legend(loc='upper right')

    # Hide any unused subplots
    for j in range(i + 1, num_subplots):
        axes[j].set_visible(False)

    # Set title and adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leavespace for suptitle

    # Save and close figure
    plt.savefig(filename)
    plt.close()

plot_hist_for_groups(group1,
                        'Residence times distribution for 3rd D value of each group (normalized by forcing period)',
                        os.path.join(image_directory1, 'res_times_group1_3rd.png'),
                        residence_times_dict_1, forcing_period)

plot_hist_for_groups(group2,
                        'Residence times distribution for 3rd D value of each group (normalized by forcing period)',
                        os.path.join(image_directory2, 'res_times_group1_3rd.png'),
                        residence_times_dict_2, forcing_period)


