"""
Run this when you already have your data downloaded and unzipped.

Date: 19 October 2015
"""
__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.1.0'

import glob
import os
import csv
import numpy as np
from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(labels_img='/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/TT/tt_mask_pad.nii'
                           , standardize=True)  # sets the atlas used
atlas = 'TT'  # label which atlas to use
stud = 'Olin'
os.chdir('/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/{0}'.format(stud))
for n in sorted(glob.glob('*[0-9].nii')):
    str_id = n[:5]  # sets the current image ID
#    masked_data = apply_mask(n, str_id+'_mask.nii')
    ts = masker.fit_transform('{0}.nii'.format(str_id))

    norm = np.corrcoef(ts.T)

    # flatten the correlation matrix
    cors = [str_id]
    for i in range(1, np.size(norm, axis=0)):
        for j in range(i+1, np.size(norm, axis=0)):
            cors.append(norm[i, j])

    # write the correlations to a CSV
    with open('{1}_{0}.csv'.format(atlas, stud), 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(cors)
    csvfile.close()
    print(str_id)
