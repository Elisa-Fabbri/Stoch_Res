import os
import joblib
import sys
import configparser
import functions as fn
import aesthetics as aes
import numpy as np
import re

binarized_trajectory_file_path = sys.argv[1]
binarized_trajectory = joblib.load(binarized_trajectory_file_path)

binarized_trajectory_file = os.path.basename(binarized_trajectory_file_path)

threshold_method = str(re.split(r'[._]', binarized_trajectory_file)[1])

ts = np.array(binarized_trajectory['ts'])

for key in binarized_trajectory.keys():
    if key == 'ts':
        continue
    residence_times = fn.find_residence_time(binarized_trajectory[key], ts)
    binarized_trajectory[key] = residence_times

output_directory = os.path.dirname(binarized_trajectory_file_path)
output_file = f'residence_times_{threshold_method}.pkl'
output_file = os.path.join(output_directory, output_file)

joblib.dump(binarized_trajectory, output_file)
