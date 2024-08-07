"""
Plotta il SNR calcolato
"""
import os
import joblib
import numpy as np
import pandas as pd

SNR_04_path = 'RK_forcing_0.4/SNR.csv'
SNR_02_path = 'RK_forcing_0.2/SNR.csv'

#Legge i file csv 

SNR_04 = pd.read_csv(SNR_04_path, header = 0)
SNR_02 = pd.read_csv(SNR_02_path, header = 0)

#Plotta la colonna SNR rispetto alla colonna key

#Float se non lo sono già
SNR_04['key'] = SNR_04['key'].astype(float)
SNR_04['SNR'] = SNR_04['SNR'].astype(float)

SNR_02['key'] = SNR_02['key'].astype(float)
SNR_02['SNR'] = SNR_02['SNR'].astype(float)

#Plot
import matplotlib.pyplot as plt

plt.plot(SNR_04['key'], SNR_04['SNR'], label = 'Forcing amplitude 0.4')
plt.plot(SNR_02['key'], SNR_02['SNR'], label = 'Forcing amplitude 0.2')

plt.xlabel('D')
plt.ylabel('SNR')
plt.title('SNR for different forcing Amplitudes')
plt.legend(fontsize = 14)

plt.savefig('immagini/SNR_forcing_amplitudes.png')
plt.close()


