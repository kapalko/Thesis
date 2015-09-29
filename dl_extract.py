"""
This python script is used to automatically download the selected data to a machine, extract it, and then move it to a
working directory.

Date: 29 September 2015
"""

__author__ = '2d Lt Kyle Palko'
__version__ = 'v0.3.1'

from nilearn import datasets
import glob
import os
import gzip

path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test'  # set where the data should be saved
sub_id = [50004, 50121, 50003]  # sets the subject ids that should be pulled simple 3
# sub_id = range(50003, 50060)  # calls all of the PITT study subjects
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
    datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline=pipeline, band_pass_filtering=True,
                             derivatives=['func_mask'], SUB_ID=n)
    # extract and rename the mask
    for name in glob.glob('*' + str(n) + '*.gz'):  # use glob to find the recently download filename
        inF = gzip.open(name, 'rb')  # opens .gz file
        outF = open('{0}_mask.nii'.format(n), 'wb')   # creates a new file using fileID as the name
        outF.write(inF.read())  # extract and write the .nii file
        inF.close()
        outF.close()
        os.remove(name)  # deletes the .nii.gz file
# at this point, I would just move to the folder, select all, and move to the data folder that you desire to house
# the data
