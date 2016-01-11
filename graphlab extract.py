__author__ = '2d Lt Kyle Palko'

from random import shuffle
from random import seed
import numpy as np
import csv
import time
import graphlab as gl

d_path = 'csv/tt_prep_cpac_nofilt_global.csv'  # desktop
# d_path = '/home/kap/Thesis/Data/csv/dos160_prep_cpac_filt_noglobal.csv'
num_runs = 1000  # number of runs to perform the classifiers

# Write results
write_coef = True  # whether or not to output the coefficients in a CSV file
write_results = True
result_title = 'tt_sex_nofilt_global'

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

    trn_x = gl.SFrame(trn_x)
    trn_y = gl.SFrame(trn_y)
    val_x = gl.SFrame(val_x)
    val_y = gl.SFrame(val_y)
    tst_x = gl.SFrame(tst_x)
    tst_y = gl.SFrame(tst_y)

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

seed(41)
j = 0
while j < num_runs:
    trn_x, trn_y, val_x, val_y, tst_x, tst_y = tvt(X, Y)
    lgc = BeginClass()
    lgc.lst()
    extractor = gl.feature_engineering.DeepFeatureExtractor()

    sf = extractor.fit_transform(trn_x)
    model = gl.logistic_classifier.create(sf, trn_y, l2_penalty=0, l1_penalty=0.4)
    predictions = model.classify(tst_x)
    results = model.evaluate(tst_y)
