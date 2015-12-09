"""
This file writes column numbers to a csv so you can compare how often columns are involved in the regularization
coefficients.
"""

__author__ = '2d Lt Kyle Palko'

import csv
import numpy as np

num_roi = 96  # number of ROIs in the atlas
col_one = []
col_two = []
for i in range(1, 96):
    for j in range(i+1, 97):
        col_one.append(i)
        col_two.append(j)

for i in range(0, np.size(col_one, axis=0)):
    with open('columns.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow((col_one[i], col_two[i]))
    csvfile.close()