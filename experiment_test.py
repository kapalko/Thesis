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
from sklearn.linear_model import LogisticRegression as lg
# from sklearn.svm import SVC as lg
# from sklearn.ensemble import AdaBoostClassifier as ADA
# from sklearn.ensemble import RandomForestClassifier as ADA
import csv
import time


start_time = time.time()
__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.1.2'

d_path = 'csv/cc200_prep_cpac_filt_noglobal.csv'  # desktop
# d_path = '/home/kap/Thesis/Data/csv/dos160_prep_cpac_filt_noglobal.csv'
num_runs = 1000  # number of runs to perform the classifiers

# Write results
write_coef = True  # whether or not to output the coefficients in a CSV file
write_results = True
result_title = 'cc200_age19_sex_noise_filt_noglobal'

id_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/ID_code.csv'

# PCA options
do_pca = False
if do_pca:
    from sklearn.decomposition import PCA
    n_pca = .85  # % of variance to keep
    acc = np.zeros((num_runs, 4))
else:
    acc = np.zeros((num_runs, 3))

# 2 degree factorial model with interactions
# make sure you have lots of memory for this one
do_full = False
if do_full:
    full_path = 'Results/tt_full.csv'

# include gender?
do_sex = True
if do_sex:
    id_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/ID_code.csv'
    s_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/sex.csv'

# create a noise variable
do_noise = True

# plot CV
cv_plot = False
if cv_plot:
    num_runs = 1
    from matplotlib import pyplot as plt

# do an age restricted model
do_age = True
if do_age:
    age_lim = 19
    a_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/age.csv'
    del_a = True # delete age from the data before running model
    min_age = False
    if min_age:
        m_age = 12


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

def resid(y_pred, y_act):
    res = y_pred - y_act #subtract predictions and actual to form residual
    MSE = sum(np.square(res))/len(res)
    return res, MSE

def modResid(model, x_data, y_data):
    y_pred = model.predict(x_data)
    res, MSE = resid(y_pred, y_data)
    return res, MSE

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

if do_sex or do_age:
    idlab = []
    sex = []
    age = []
    with open(id_path, 'rb') as f:
        spamreader = csv.reader(f, delimiter=',')
        for row in spamreader:
            idlab.append(row)
    f.close()

    if do_sex: s = np.genfromtxt(s_path, delimiter=',')
    if do_age: a = np.genfromtxt(a_path, delimiter=',')
    # find DX by matching the rows
    for subid in (x[0] for x in data):
        d = idlab.index(['{0}'.format(int(subid))])
        if do_sex: sex.append(s[d]-1)
        if do_age: age.append(a[d])

if do_sex:
    data = np.column_stack((data, sex))
    del idlab
    del sex
    del s

if do_age:
    data = np.column_stack((data, age))
    del age
    del a

    if min_age: data = data[np.logical_not(np.logical_or(data[:, -1] >= age_lim, data[:, -1] < m_age))] # keep only those between age limits
    else: data = data[np.logical_not(data[:, -1] >= age_lim)]  # keep only those under age limit

    if del_a: data = np.delete(data, np.s_[-1:], 1)  # take age out of data

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
print result_title

while j < num_runs:

    trn_x, trn_y, val_x, val_y, tst_x, tst_y = tvt(X, Y)

    if do_pca:
        r = PCA(n_components=n_pca)
        trn_x = r.fit_transform(trn_x)
        val_x = r.transform(val_x)
        tst_x = r.transform(tst_x)
        acc[j, 3] = np.size(trn_x, axis=1)

    lgc = BeginClass()
    lgc.lst()

    # sr0 = np.zeros((10, 4))
    # a = 0
    for c in np.linspace(.0001, 5, 50):
        lgr = lg(penalty='l1', C=c)
    #     lgr = lg(C=c, kernel='linear')
    # for c in [1000]:
    #     lgr = ADA(n_estimators=c)
        lgc.appen(model=lgr, param=c, trnx=trn_x, trny=trn_y, valx=val_x, valy=val_y)
    ############
        # # Use for printing MSE figure for CV
        # lgr.fit(trn_x, trn_y)
        # sr0[a, 1] = modResid(lgr, trn_x, trn_y)[1] #returns the MSE
        # sr0[a, 2] = modResid(lgr, val_x, val_y)[1]
        # sr0[a, 3] = modResid(lgr, tst_x, tst_y)[1]
        # sr0[a, 0] = c
        # a += 1

    lgc.locate()
    c = lgc.param[lgc.plac]
    lgr = lg(penalty='l1', C=c)
    # lgr = lg(C=c, kernel='linear')
    # lgr = ADA(n_estimators=c)
    lgc.update(lgr, trnx=trn_x, trny=trn_y, tstx=tst_x, tsty=tst_y)


    coef[:np.size(lgr.coef_), j] = lgr.coef_
    acc[j, 0] = lgc.acc
    acc[j, 1] = np.count_nonzero(coef[:, j])
    acc[j, 2] = c


    print 'Accuracy: {0}'.format(lgc.acc)
    print 'c={0}'.format(c)
    print(np.count_nonzero(coef[:, j]))
    print j

    j += 1

    # if cv_plot:
    #     fig = plt.figure()
    #     plt.plot(sr0[:, 0], sr0[:, 1], c='r', label='Training MSE')
    #     plt.plot(sr0[:, 0], sr0[:, 2], c='g', label='Validation MSE')
    #     plt.plot(sr0[:, 0], sr0[:, 3], c='b', label='Testing MSE')
    #     plt.legend(loc='lower right')
    #     fig.title('Cross Validation Results')
    #     plt.xlabel('Parameter (C)')
    #     plt.ylabel('MSE')
    #     plt.savefig('MSEPlot.png') #, bbox_inches = 'tight'
    #     plt.close()

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
