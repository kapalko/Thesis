__author__ = '2d Lt Kyle Palko'

import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt('/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/Test/ABIDE_pcp/cpac/filt_noglobal/Pitt_0050003_rois_tt.1D', skip_header=1)
c = np.corrcoef(data.T)
plt.figure(figsize=(10, 10))
plt.imshow(c, interpolation='none')
plt.colorbar(shrink=.5)
plt.title('Correlation Matrix of ROIs of Subject 50003', fontsize=20)
plt.xlabel('ROI')
plt.ylabel('ROI')
