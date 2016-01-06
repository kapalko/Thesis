"""
This python file is used to run the experimental tests. There are several options available for the user. Please look at
the initial variables to change what you want.

This file perfoms several iterations of Cross Validation to select the best hyperparameter for use in Logistic
Regression. The classification method can be easily changed by the user with just a few changes in code.

Parameters:
    d_path: The path to the FNC values
    num_runs: number of experimental runs, each with a different random TVT allocation
    write_coef: Select if you would like to keep the coefficient values in a CSV
    write_results: Select if you would like to keep the results in a CSV
    result_title: The name of the resulting CSVs
    do_pca: Whether or not to perform PCA on the data before classifying
    n_pca: If integer, the number of columns to keep. If (0,1), the percentage of variance to keep.
    do_full: Create a full, 2 degree factorial model with interactions (only uses a select number of columns).
    full_path: The path to a csv that contains the important columns to use in full model.
    do_noise: If true, a Gaussian normal vector is added to the data to compare how often it is selected.

Date: 05 January 2016
"""
from random import shuffle
from random import seed
import numpy as np
from sklearn.metrics import confusion_matrix as cm
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as lg
import csv
from sklearn.decomposition import PCA
# from sklearn.ensemble import AdaBoostClassifier as ADA
import time

start_time = time.time()
__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.1.2'

d_path = 'csv/TT_prep_cpac_filt_noglobal.csv'  # desktop
# d_path = '/home/kap/Thesis/Data/csv/dos160_prep_cpac_filt_noglobal.csv'
num_runs = 1000  # number of runs to perform the classifiers

# Write results
write_coef = True  # whether or not to output the coefficients in a CSV file
write_results = True
result_title = 'TT_full'

# PCA options
do_pca = False
n_pca = .9  # % of variance to keep

# 2 degree factorial model with interactions
# make sure you have lots of memory for this one
do_full = True
full_path = 'Results/tt_full.csv'

# create a noise variable
do_noise = False


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


class BeginClass():

    def lst(self):
        self.val_score = []
        self.param = []

    def appen(self, model, param, trnx, trny, valx, valy):
        model.fit(trnx, trny)
        self.val_score.append(model.score(valx, valy))
        self.param.append(param)

    def locate(self):
        self.plac = (np.where(max(self.val_score) == self.val_score))[0][0]

    def update(self, model, trnx, trny, tstx, tsty):
        model.fit(trnx, trny)
        self.acc = model.score(tstx, tsty)
        prediction = model.predict(tstx)  # predict the outcomes
        self.con = cm(tsty, prediction) # creates confusion matrix


# get the data
data = np.genfromtxt(d_path, delimiter=',')

# remove rows that have NaN values (not ideal)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
data = sorted(data, key=lambda x: x[0])
Y = np.array([x[1]-1 for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del data

if do_full:
    c = 0
    # interact = np.zeros((np.size(X, axis=0), (np.size(X, axis=1)*(np.size(X, axis=1)-1))/2))  # initialize (roughly 8,731,396,800 cells)
    # for i in range(0, np.size(X, axis=1)-1):
    #     print i
    #     for j in range(i+1, np.size(X, axis=1)):
    #         interact[:, c] = (X[:, i]*X[:, j])
    #         c += 1

    col = np.genfromtxt(full_path, delimiter=',')
    col = sorted(col)
    interact = np.zeros((np.size(X, axis=0), (np.size(col)*(np.size(col)-1))/2))
    for i in range(0, np.size(col)-1):
        for j in range(i+1, np.size(col)):
            interact[:, c] = X[:, col[i]]*X[:, col[j]]
            c += 1

    X = np.append(X, np.square(X), axis=1)  # add the squares
    X = np.append(X, interact, axis=1)
    print 'Completed Full Model'


if do_noise:
    from sklearn.preprocessing import MinMaxScaler as ss
    stan = ss(feature_range=(-1, 1))

    x_norm = np.random.randn(np.size(X, axis=0), 1)
    X = np.column_stack((X, x_norm))
    X = stan.fit_transform(X)

# build train test validate sets
# seed(41)
j = 0
coef = np.zeros((np.size(X, axis=1), num_runs))
if do_pca:
    acc = np.zeros((num_runs, 4))
else:
    acc = np.zeros((num_runs, 3))

while j < num_runs:

    trn_x, trn_y, val_x, val_y, tst_x, tst_y = tvt(X, Y)

    if do_pca:
        r = PCA(n_components=.9)
        trn_x = r.fit_transform(trn_x)
        val_x = r.transform(val_x)
        tst_x = r.transform(tst_x)
        acc[j, 3] = np.size(trn_x, axis=1)

    lgc = BeginClass()
    lgc.lst()

    for c in np.linspace(0.0001, 4, 30):
        lgr = lg(penalty='l1', C=c)
    # for c in range(1, 100, 5):
    #     lgr = ADA(n_estimators=c)
        lgc.appen(model=lgr, param=c, trnx=trn_x, trny=trn_y, valx=val_x, valy=val_y)

    lgc.locate()
    c = lgc.param[lgc.plac]
    lgr = lg(penalty='l1', C=c)
    # lgr = ADA(n_estimators=c)
    lgc.update(lgr, trnx=trn_x, trny=trn_y, tstx=tst_x, tsty=tst_y)

    coef[:np.size(lgr.coef_), j] = lgr.coef_
    acc[j, 0] = lgc.acc
    acc[j, 1] = np.count_nonzero(coef[:, j])
    acc[j, 2] = c

    print 'Accuracy: {0}'.format(lgc.acc)
    print(np.count_nonzero(coef[:, j]))
    print j

    j += 1

if write_coef:
    for i in range(0, np.size(coef, axis=0)):
        with open('{0}_coef_results.csv'.format(result_title), 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow((coef[i, :]))
        csvfile.close()

if write_results:
    for i in range(0, num_runs):
        with open('{0}_exp_results.csv'.format(result_title), 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow((acc[i, :]))
        csvfile.close()

end_time = time.time()-start_time  # seconds
print 'Run time (hours): {0}'.format(end_time/3600)