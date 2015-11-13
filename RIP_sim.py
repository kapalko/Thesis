__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.0.1'

import numpy as np
from sklearn.preprocessing import normalize as norm

# constants
csv_path = 'tt_cpac_filt_noglobal.csv'

# get the data
data = np.genfromtxt(csv_path, delimiter=',')
# remove rows that have NaN values (not ideal but IDGAF yet)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
Y = np.array([x[1]-1 for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del x
del data

x_norm = norm(X, axis=0)
