"""
This python file is used to compare the correlations of the FNC data.

Date: 06 December 15
"""
import numpy as np
from matplotlib import pyplot as plt
import csv
import os

# constants
dpi = 1000  # pixels per inch (change for laptop or slower computers)

# d_path = '/home/kap/Thesis/Data/csv/TT_prep_cpac_filt_noglobal.csv'  # laptop
d_path = 'csv/TT_prep_cpac_filt_noglobal.csv'  # desktop

# get the data
data = np.genfromtxt(d_path, delimiter=',')

# remove rows that have NaN values (not ideal but IDGAF yet)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
data = sorted(data, key=lambda x: x[0])
Y = np.array([x[1]-1 for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del x
del data

corr = np.corrcoef(X.T)

# os.chdir('/home/kap/Thesis/Data/csv/')

plt.figure(figsize=(10, 10))
plt.imshow(corr, interpolation='nearest')
plt.title('Correlations among FNC values')
plt.savefig('FNC_Correlation.png', dpi=dpi)
plt.close()


# # write the correlations to a CSV
# for i in range(0, np.size(corr, axis=1)):
#     with open('FNC_corrs.csv', 'ab') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=',')
#         spamwriter.writerow(corr[i, :])
#     csvfile.close()
