"""
This file is used to perform the Reverse Isometry Property simulations in an attempt to find the probability that
the RIP condition holds for the reduced model.

Date: 17 November 15
"""

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.1.0'

import numpy as np
from sklearn.preprocessing import normalize as norm
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler as ss

# constants
# csv_path = 'tt_cpac_filt_noglobal.csv'  # desktop
csv_path = '/home/kap/Thesis/Data/csv/dos160_prep_cpac_filt_noglobal.csv'  # laptop
num_runs = 10000
max_s = 2000
run = True
b = 1

def delt(beta, vect):
    # use optimization to find the minimum positive delta with constraints
    # (1-d)||b||_2^2 <= ||Xb||_2^2 <= (1+d)||b||_2^2
    return np.abs(np.sum(np.square(vect*beta))-1)


# get the data
data = np.genfromtxt(csv_path, delimiter=',')
# remove rows that have NaN values (not ideal but IDGAF yet)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
Y = np.array([x[1]-1 for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del x
del data
stan = ss()
x_norm = norm(stan.fit_transform(X.astype('float')), axis=0)
n_feat = np.size(X, axis=1)

results = np.zeros((num_runs, max_s))
d_quant = np.zeros((max_s, 4))

while run and b <= max_s:
    print b

    for i in range(0, num_runs):
        beta = np.repeat(1/np.sqrt(b), b)  # creates vector of betas
        z = np.random.random_integers(0, n_feat-1, b)  # gets random integers for columns from data
        vect = x_norm[:, z]  # stores normal x values in a new array to multiply by betas
        results[i, b-1] = delt(beta, vect)

    d_quant[b-1, 0] = b
    d_quant[b-1, 1] = np.percentile(results[:, b-1], 10)
    d_quant[b-1, 2] = np.percentile(results[:, b-1], 50)
    d_quant[b-1, 3] = np.percentile(results[:, b-1], 90)
    b += 1
fig = plt.figure()
plt.plot(d_quant[:, 0], d_quant[:, 1], c='r', label='.1 Quantile')
plt.plot(d_quant[:, 0], d_quant[:, 2], c='g', label='.5 Quantile')
plt.plot(d_quant[:, 0], d_quant[:, 3], c='b', label='.9 Quantile')
plt.legend(loc='lower left')
plt.xlabel('Beta')
plt.ylabel('delta')
plt.savefig('test_plot.png')
plt.close()
print('End')
