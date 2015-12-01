"""
This python file is used to run the experimental tests.

Date: 25 November 15
"""
from random import shuffle
from random import seed
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as lg
import csv
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier as ADA

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.0.8'

# d_path = 'csv/dos160_prep_cpac_filt_noglobal.csv'
d_path = '/home/kap/Thesis/Data/csv/dos160_prep_cpac_filt_noglobal.csv'
num_runs = 10  # number of runs to perform the classifiers
write_coef = False  # whether or not to output the coefficients in a CSV file
write_results = True

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

# remove rows that have NaN values (not ideal but IDGAF yet)
data = data[~np.isnan(data).any(axis=1)]  # from stack overflow: https://bit.ly/1QhfcmZ
data = sorted(data, key=lambda x: x[0])
Y = np.array([x[1]-1 for x in data])  # y values in the second column
X = np.array([x[2:] for x in data])
del x
del data

# build train test validate sets
seed(41)
j = 0
coef = np.zeros((np.size(X, axis=1), num_runs))
acc = np.zeros((num_runs, 2))
while j < num_runs:

    trn_x, trn_y, val_x, val_y, tst_x, tst_y = tvt(X, Y)

    # svm
    # initialize
    # svmc = BeginClass()
    # svmc.lst()
    #
    # for c in np.linspace(.0001, 3, 30):
    #     svmr = SVC(C=c, kernel='linear')
    #     svmc.appen(model=svmr, param=c, trnx=trn_x, trny=trn_y, valx=val_x, valy=val_y)
    #
    # svmc.locate()
    # c = svmc.param[svmc.plac]
    # svmr = SVC(C=c, kernel='linear')
    # svmc.update(svmr, trnx=trn_x, trny=trn_y, tstx=tst_x, tsty=tst_y)
    #
    # print c
    # print 'Accuracy: {0}'.format(svmc.acc)
    # print 'Confusion matrix: '
    # print svmc.con
    r = PCA(n_components=.9)
    rtrn = r.fit_transform(trn_x)
    rval = r.transform(val_x)
    rtst = r.transform(tst_x)

    lgc = BeginClass()
    lgc.lst()

    # for c in np.linspace(0.0001, 5, 30):
        # lgr = lg(penalty='l1', C=c)
    for c in range(1, 100, 5):
        lgr = ADA(n_estimators=c)
        lgc.appen(model=lgr, param=c, trnx=trn_x, trny=trn_y, valx=val_x, valy=val_y)

    lgc.locate()
    c = lgc.param[lgc.plac]
    # lgr = lg(penalty='l1', C=c)
    lgr = ADA(n_estimators=c)
    lgc.update(lgr, trnx=trn_x, trny=trn_y, tstx=tst_x, tsty=tst_y)
    coef[:, j] = lgr.coef_
    print 'Untransformed'
    print c
    print 'Accuracy: {0}'.format(lgc.acc)
    print 'Confusion matrix: '
    print lgc.con
    print(np.count_nonzero(coef[:, j]))


    lgc = BeginClass()
    lgc.lst()
    # for c in np.linspace(0.0001, 5, 30):
    #     lgr = lg(penalty='l1', C=c)
    for c in range(1, 100, 5):
        lgr = ADA(n_estimators=c)
        lgc.appen(model=lgr, param=c, trnx=rtrn, trny=trn_y, valx=rval, valy=val_y)

    lgc.locate()
    c = lgc.param[lgc.plac]
    # lgr = lg(penalty='l1', C=c)
    lgr = ADA(n_estimators=c)
    lgc.update(lgr, trnx=rtrn, trny=trn_y, tstx=rtst, tsty=tst_y)
    rcoef = lgr.coef_
    acc[j, 0] = lgc.acc
    acc[j, 1] = np.count_nonzero(rcoef)
    print 'transformed'
    print c
    print 'Accuracy: {0}'.format(lgc.acc)
    print 'Confusion matrix: '
    print lgc.con
    print(np.count_nonzero(rcoef))
    j += 1

if write_coef:
    for i in range(0, np.size(coef, axis=0)):
        with open('coef.csv'.format(j), 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow((coef[i, :]))
        csvfile.close()

if write_results:
    for i in range(0, num_runs):
        with open('exp_results.csv'.format(j), 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow((acc[i, :]))
        csvfile.close()