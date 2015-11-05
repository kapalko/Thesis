"""
This python file is used to run the experimental tests.

Date: 05 November 15
"""

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.0.1'

from random import shuffle
from random import seed
import numpy as np


def tvt(x_data, y_data):

    list1_shuf = []  # create empty lists to fill
    list2_shuf = []
    sz = len(y_data)  # find size of incoming data
    index_shuf = range(sz)
    shuffle(index_shuf)  # shuffle index
    for i in index_shuf:
        list1_shuf.append(x_data[i, :])  # add to new list
        list2_shuf.append(y_data[i])
    list1_shuf = np.array(list1_shuf)  # create an array from the lists
    list2_shuf = np.array(list2_shuf)

    trn_size = np.ceil(sz/2)  # training set is half of full set
    val_size = np.ceil((sz - trn_size)/2)
    tst_size = sz - trn_size - val_size
    val_size = sz - tst_size

    trn_x = list1_shuf[:trn_size, :]  # partition data into train, validate, and testing sets
    trn_y = list2_shuf[:trn_size]
    val_x = list1_shuf[trn_size:val_size, :]
    val_y = list2_shuf[trn_size:val_size]
    tst_x = list1_shuf[val_size:, :]
    tst_y = list2_shuf[val_size:]
    return trn_x, trn_y, val_x, val_y, tst_x, tst_y  # return the data

# get the data
data = np.genfromtxt('csv/tt_cpac_filt_noglobal.csv', delimiter=',')
# remove rows that have NaN values (not ideal but IDGAF yet)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
Y = np.array([x[1] for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del x
del data

# build train test validate sets
seed(41)
trn_x, trn_y, val_x, val_y, tst_x, tst_y = tvt(X, Y)
