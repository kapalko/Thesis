"""
This python script is used to automatically download the selected data to a machine, extract it, and then move it to a
working directory.

Version 0.4.0 includes flattening the correlation matrix and writing to a CSV file.

Date: 19 October 2015
"""

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.4.1'
from nilearn import datasets
import glob
import os
import gzip
import numpy as np
import csv

path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test'  # set where the data should be saved
# sub_id = [50060]  # sets the subject ids that should be pulled simple 3
# sub_id = range(50003, 50061)  # calls all of the PITT study subjects
pipeline = 'cpac'  # define the pipeline used to preprocess the data
derivative = 'func_preproc'  # define what data should be pulled

# set the directory where the data is stored so the script can find and rename the data
os.chdir('/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test/ABIDE_pcp/cpac/filt_noglobal/')

for n in sub_id:
    # download the fMRI image
    datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline=pipeline, band_pass_filtering=True,
                             derivatives=[derivative], SUB_ID=n)  # fetch the data based on the subject ID
    # extract and rename the image file
    for name in glob.glob('*' + str(n) + '*.gz'):  # use glob to find the recently download filename
        inF = gzip.open(name, 'rb')  # opens .gz file
        outF = open('{0}.nii'.format(n), 'wb')   # creates a new file using fileID as the name
        outF.write(inF.read())  # extract and write the .nii file
        inF.close()
        outF.close()
        os.remove(name)  # deletes the .nii.gz file

    # download the image's corresponding mask
#    datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline=pipeline, band_pass_filtering=True,
#                             derivatives=['func_mask'], SUB_ID=n)
#    # extract and rename the mask
#    for name in glob.glob('*' + str(n) + '*.gz'):  # use glob to find the recently download filename
#        inF = gzip.open(name, 'rb')  # opens .gz file
#        outF = open('{0}_mask.nii'.format(n), 'wb')   # creates a new file using fileID as the name
#        outF.write(inF.read())  # extract and write the .nii file
#        inF.close()
#        outF.close()
#        os.remove(name)  # deletes the .nii.gz file
# at this point, I would just move to the folder, select all, and move to the data folder that you desire to house
# the data

# from nilearn.masking import apply_mask
from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(labels_img='/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/TT/tt_mask_pad.nii'
                           , standardize=True)  # sets the atlas used
atlas = 'TT'  # label which atlas to use
stud = 'Pitt'
os.chdir('/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Data/Pitt')
for n in sorted(glob.glob('*[0-9].nii')):
    str_id = n[:5]  # sets the current image ID
#    masked_data = apply_mask(n, str_id+'_mask.nii')    
    ts = masker.fit_transform('{0}.nii'.format(str_id))
    
    norm = np.corrcoeff(ts.t)
    
    # flatten the correlation matrix
    cors = [str_id]
    for i in range(1, np.size(norm, axis=0)):
        for j in range (i+1, np.size(norm, axis=0)):
            cors.append(norm[i,j])
     
    # write the correlations to a CSV
    with open('{1}_{0}.csv'.format(atlas, stud), 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(cors)
    csvfile.close()
