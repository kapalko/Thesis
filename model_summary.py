__author__ = '2d Lt Kyle Palko'

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

d_path = 'csv/cc200_prep_cpac_filt_noglobal.csv'  # desktop
write_coef = True  # whether or not to output the coefficients in a CSV file
write_results = True
result_title = 'cc200_testing'
do_sex = True
if do_sex:
    id_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/ID_code.csv'
    s_path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/sex.csv'

do_age = False
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

#####################
#####################

seed(41)
trn_x, trn_y, val_x, val_y, tst_x, tst_y = tvt(X, Y)
coef = np.zeros((np.size(X, axis=1), 1))
acc = np.zeros((1, 3))

lgc = BeginClass()
lgc.lst()

# sr0 = np.zeros((10, 4))
# a = 0
for c in np.linspace(.0001, 5, 50):
    lgr = lg(penalty='l1', C=c)
#     lgr = lg(C=c, kernel='linear')
# for c in range(1, 300, 10):
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


coef[:np.size(lgr.coef_), 0] = lgr.coef_
acc[0, 0] = lgc.acc
acc[0, 1] = np.count_nonzero(coef[:, 0])
acc[0, 2] = c

print 'Accuracy: {0}'.format(lgc.acc)
print 'c={0}'.format(c)
print(np.count_nonzero(coef[:, 0]))


#############
#############


for i in range(0, np.size(coef, axis=0)):
    with open('{0}_coef_results.csv'.format(result_title), 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow((coef[i][0], ''))
    csvfile.close()
print 'complete coef'

data = np.column_stack((trn_y, trn_x))

aut = data[np.logical_not(data[:, 0] == 1)]
con = data[np.logical_not(data[:, 0] == 0)]
aut_sum = np.average(aut[:, 1:], axis=0)
con_sum = np.average(con[:, 1:], axis=0)

comb = np.column_stack((aut_sum.T, con_sum.T))

for i in range(0, np.size(con_sum)):
    with open('cc200mod_sum.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(comb[i, :])
    csvfile.close()
print 'complete'