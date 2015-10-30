"""
Extracts the FNC data from the images (either .gz or ,nii) and puts them into a CSV file for use. Note you need to
identify where the labels are ('lab'). You need two CSV files, the subject IDs and the diagnosis file. Sub_ID.csv and
DX.csv.

Download the csv files here:
https://www.amazon.com/clouddrive/share/eAMBKfrbBdCRDfmreADaF4oGoZ4ltJGAIWz9I0TtPZT?ref_=cd_ph_share_link_copy

Date: 29 October 15
"""

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.0.2'

import glob
import os
import gzip
import csv
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
import time

start_time = time.time()

# local variables and paths #
path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/'  # working directory
mask = 'masks/'  # location of mask
pipe = 'cpac'
filt = 'filt_noglobal'
stud = 'Test/ABIDE_pcp/{0}/{1}/'.format(pipe, filt)  # location that download happened
# stud = 'Data/'
lab = path + 'Data/'  # location of CSV files for labeling

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
for name in sorted(glob.glob('*.gz')):  # use glob to find the recently download filename
    subid = name.split('_00')  # separates the keywords to extract the ID number
    subid = subid[1][:5]  # extract ID

    inF = gzip.open(name, 'rb')  # opens .gz file
    outF = open('{0}.nii'.format(subid), 'wb')   # creates a new file using fileID as the name
    outF.write(inF.read())  # extract and write the .nii file
    inF.close()
    outF.close()
    os.remove(name)  # deletes the .nii.gz file

os.chdir(path+mask)
for msk in sorted(glob.glob('*')):
    mask_name = msk[:-4]
    masker = NiftiLabelsMasker(labels_img=msk, standardize=True)  # sets the atlas used

    os.chdir(path+stud)
    for name in sorted(glob.glob('*[0-9].nii')):
        subid = name[:5]
        # extract time series data
        ts = masker.fit_transform(name)  # masks must be in same directory as data
        norm = np.corrcoef(ts.T)

        # find DX by matching the rows
        cors = [subid]
        d = idlab.index([subid])
        cors.append(dxlab[d][0])
        # flatten the correlation matrix
        for i in range(1, np.size(norm, axis=0)):
            for j in range(i+1, np.size(norm, axis=0)):
                cors.append(norm[i, j])

        # write the correlations to a CSV
        with open('{0}_{1}_{2}.csv'.format(mask_name, pipe, filt), 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(cors)
        csvfile.close()
        print(subid)

    print ('Complete with {0}'.format(mask_name))

end_time = (time.time()-start_time)/60

print ('Completed in {0} minutes'.format(end_time))
