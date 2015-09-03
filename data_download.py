"""
This file will automatically pull the data from the storage on the cloud. Afterwards, you must extract and put the files
into your working directory.

See https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_abide_pcp.html#nilearn.datasets.fetch_abide_pcp
for instructions
"""

__author__ = 'kap'

from nilearn import datasets
ab_img_one = datasets.fetch_abide_pcp(n_subjects=2, pipeline='cpac', band_pass_filtering=True, derivatives=['func_preproc'], SUB_ID=[50003])
ab_img_two = datasets.fetch_abide_pcp(n_subjects=2, pipeline='cpac', band_pass_filtering=True, derivatives=['func_preproc'], SUB_ID=[50004])

ab_mask_one = datasets.fetch_abide_pcp(n_subjects=2, pipeline='cpac', band_pass_filtering=True, derivatives=['func_mask'], SUB_ID=[50003])
ab_mask_two = datasets.fetch_abide_pcp(n_subjects=2, pipeline='cpac', band_pass_filtering=True, derivatives=['func_mask'], SUB_ID=[50003])
