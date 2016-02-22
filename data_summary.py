__author__ = '2d Lt Kyle Palko'

import numpy as np
import csv

do_age = True
do_sex = False

d_path = 'csv/cc200_prep_cpac_filt_noglobal.csv'
id_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/ID_code.csv'
# get the data
data = np.genfromtxt(d_path, delimiter=',')

# remove rows that have NaN values (not ideal)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
data = sorted(data, key=lambda x: x[0])
data = np.array(data)

if do_age:
    age_lim = 19
    a_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/age.csv'
    del_a = True # delete age from the data before running model
    min_age = False
    if min_age:
        m_age = 12

if do_sex or do_age:
    idlab = []
    sex = []
    age = []
    with open(id_path, 'rb') as f:
        spamreader = csv.reader(f, delimiter=',')
        for row in spamreader:
            idlab.append(row)
    f.close()

    if do_age: a = np.genfromtxt(a_path, delimiter=',')
    # find DX by matching the rows
    for subid in (x[0] for x in data):
        d = idlab.index(['{0}'.format(int(subid))])
        if do_age: age.append(a[d])

if do_age:
    data = np.column_stack((data, age))
    del age
    del a

    if min_age: data = data[np.logical_not(np.logical_or(data[:, -1] >= age_lim, data[:, -1] < m_age))] # keep only those between age limits
    else: data = data[np.logical_not(data[:, -1] >= age_lim)]  # keep only those under age limit

    if del_a: data = np.delete(data, np.s_[-1:], 1)  # take age out of data


aut = data[np.logical_not(data[:, 1] == 2)]
con = data[np.logical_not(data[:, 1] == 1)]
aut_sum = np.average(aut[:, 2:], axis=0)
con_sum = np.average(con[:, 2:], axis=0)

comb = np.column_stack((aut_sum.T, con_sum.T))

for i in range(0, np.size(con_sum)):
    with open('cc200_aut_19_sum.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(comb[i, :])
    csvfile.close()