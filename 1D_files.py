__author__ = '2d Lt Kyle Palko'

import glob
import os
import gzip
import csv
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets

path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test'  # set where the data should be saved
pipeline = 'cpac'  # define the pipeline used to preprocess the data
derivative = 'rois_tt'  # define what data should be pulled

datasets.fetch_abide_pcp(data_dir=path, pipeline=pipeline, band_pass_filtering=True, global_signal_regression=False,
                         derivatives=[derivative])

# local variables and paths #
path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/'  # working directory
pipe = 'cpac'
filt = 'filt_noglobal'
stud = 'Test/ABIDE_pcp/{0}/{1}/'.format(pipe, filt)  # location that download happened
# stud = 'Data/'
lab = path + 'Data/'  # location of CSV files for labeling
mask_name = 'TT_prep'

# build two lists of strings from CSV files to use to match the subjects and their diagnosis
idlab = []  # subject IDs
dxlab = []  # subject diagnosis
with open(lab+'ID_code.csv', 'rb') as f:
    spamreader = csv.reader(f, delimiter=',')
    for row in spamreader:
        idlab.append(row)
f.close()

with open(lab+'DX.csv', 'rb') as f:
    spamreader = csv.reader(f, delimiter=',')
    for row in spamreader:
        dxlab.append(row)
f.close()

# extract and rename the image file
os.chdir(path+stud)  # set path of data
for name in sorted(glob.glob('*')):
    subid = name.split('_00')  # separates the keywords to extract the ID number
    subid = subid[1][:5]  # extract ID
    ts = np.genfromtxt(name, skip_header=1)
    r = np.corrcoef(ts.T)

    # find DX by matching the rows
    cors = [subid]
    d = idlab.index([subid])
    cors.append(dxlab[d][0])
    # flatten the correlation matrix
    for i in range(1, np.size(r, axis=0)):
        for j in range(i+1, np.size(r, axis=0)):
            cors.append(r[i, j])

    # write the correlations to a CSV
    with open('{0}_{1}_{2}.csv'.format(mask_name, pipe, filt), 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(cors)
    csvfile.close()
    print(subid)
