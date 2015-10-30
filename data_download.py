__author__ = '2d Lt Kyle Palko'

from nilearn import datasets
import time

start = time.time()
path = '/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test'  # set where the data should be saved
pipeline = 'cpac'  # define the pipeline used to preprocess the data
derivative = 'func_preproc'  # define what data should be pulled
stud = '/ABIDE_pcp/cpac/filt_noglobal/e806a9ef657f316b760441d3649f7cb6'

datasets.fetch_abide_pcp(data_dir=path, pipeline=pipeline, band_pass_filtering=True, derivatives=[derivative])
