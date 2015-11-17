"""
This file is used to perform the Reverse Isometry Property simulations in an attempt to find the probability that
the RIP condition holds for the reduced model.

Date: 15 November 15
"""

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.0.2'

import numpy as np
from sklearn.preprocessing import normalize as norm
import random

# constants
csv_path = 'tt_cpac_filt_noglobal.csv'
num_runs = 10
max_s = 10
run = True
b = 1

def delt(beta, vect):
    # use optimization to find the minimum positive delta with constraints
    # (1-d)||b||_2^2 <= ||Xb||_2^2 <= (1+d)||b||_2^2

    return d

# get the data
data = np.genfromtxt(csv_path, delimiter=',')
# remove rows that have NaN values (not ideal but IDGAF yet)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
Y = np.array([x[1]-1 for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del x
del data
x_norm = norm(X, axis=0)
n_feat = np.size(X, axis=1)

results = np.zeros(num_runs, max_s)

while run and b <= max_s:
    print b

    for i in range(0, num_runs):
        beta = np.repeat(1/np.sqrt(b), b)  # creates vector of betas
        z = np.random.random_integers(0, n_feat-1, b)  # gets random integers for columns from data
        vect = x_norm[:, z]  # stores normal x values in a new array to multiply by betas
        results[i, b] = delt(beta, vect)

