__author__ = '2d Lt Kyle Palko'

import numpy as np
from nilearn import plotting as plt

# import Pitt 0050013, a young autistic subject
ts = np.genfromtxt('/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/C200/ABIDE_pcp/cpac/filt_noglobal/'
                   'Pitt_0050013_rois_cc200.1D', skip_header=1)
cor = np.corrcoef(ts.T)

# create matrix
x = np.zeros((200, 200))
for i in range(0, np.size(x, axis=0)):
    x[i][i] = 1

a = np.genfromtxt('cc200_roi.csv', delimiter=',')
row = np.array([c[0] for c in a])
col = np.array([c[1] for c in a])
del a

for i in range(0, 10):
    r = row[i]
    c = col[i]
    x[r, c] = cor[r, c]
    x[c, r] = cor[c, r]

node_coords = np.genfromtxt('cc200_lab_coord.csv', delimiter=',')
plt.plot_connectome(x, node_coords, output_file='corrtest_1.png', node_size=1)