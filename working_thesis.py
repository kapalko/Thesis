"""
This python file serves as the working file for my thesis. I intend to keep clean code here and use the other files as
drafts.

For documentation, see: https://nilearn.github.io/manipulating_visualizing/manipulating_images.html#loading-data
"""

__author__ = '2d Lt Kyle Palko'

# import modules
from nilearn import datasets
from nilearn import plotting as pltt
from matplotlib import pyplot as plt

# Plot the atlas. Different Atlases provide different regions of interest (ROIs). Some ROIs are larger than others. The
# TT and AAL atlases are much smaller than the CC400. These atlases must be downloaded beforehand from
# https://preprocessed-connectomes-project.github.io/abide/Pipelines.html#regions_of_interest
pltt.plot_roi('tt_mask_pad.nii', output_file='tt_roi_plot')
pltt.plot_roi('aal_mask_pad.nii', output_file='aal_roi_plot')
pltt.plot_roi('CC400.nii', output_file='cc400_roi_plot')
print('Completed brain ROI Images')

# download the data (only two images in this case)
path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test'  # set where the data should be saved

ab_img_one = datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline='cpac', band_pass_filtering=True,
                                      derivatives=['func_preproc'], SUB_ID=[50003])
ab_img_two = datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline='cpac', band_pass_filtering=True,
                                      derivatives=['func_preproc'], SUB_ID=[50004])

ab_mask_one = datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline='cpac', band_pass_filtering=True,
                                       derivatives=['func_mask'], SUB_ID=[50003])
ab_mask_two = datasets.fetch_abide_pcp(data_dir=path, n_subjects=1, pipeline='cpac', band_pass_filtering=True,
                                       derivatives=['func_mask'], SUB_ID=[50003])

my_data = ['pitt3.nii', 'pitt4.nii']  # have to rename the two files that were downloaded

# apply mask to the fMRI images. The mask is the regions of the image that will be extracted for use.
from nilearn.masking import apply_mask
masked_data = apply_mask(my_data[0], 'pitt3mask.nii')  # just mask the first image (fMRI, mask)
print('Completed Masking')

# Use the atlas to extract image data from the masked fMRI images
from nilearn.input_data import NiftiLabelsMasker
# masker = NiftiLabelsMasker(labels_img='tt_mask_pad.nii', standardize=True)
# masker = NiftiLabelsMasker(labels_img='aal_mask_pad.nii', standardize=True)
masker = NiftiLabelsMasker(labels_img='CC400.nii', standardize=True)  # sets the atlas that will be used to extract
time_series = masker.fit_transform(my_data[0])  # extract time series data from the fMRI image

# Use the time series data to find correlations between the ROIs and plot matrix
import numpy as np
correlation_matrix = np.corrcoef(time_series.T)
plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, interpolation='nearest')

plt.show()
print('Completed Program')
