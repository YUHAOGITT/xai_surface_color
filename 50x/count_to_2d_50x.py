# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:16:24 2020

@author: Yuhao
"""

import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import Counter
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import felzenszwalb, slic, quickshift
from scipy.io import savemat

sd = 1
region = 'B'
sign = 'nega'
num_s= 300

with open('sq600_50x_'+region+'_'+sign+'_count_sd'+str(sd)+'_'+str(num_s)+'.data', 'rb') as fp:
    new_count = pickle.load(fp)
count = Counter(new_count)
idx = sorted(count.keys())
y = [count[i] for i in idx]
plt.bar(idx,y,width = 5,linewidth = 8)  
ymax = max(y)
idx_2 = [i for i, j in enumerate(y) if j == ymax]
for i in idx_2:
     plt.gca().text(idx[i], y[i], idx[i], color='b',fontsize = 8)
     print(idx[i])
     
y2 = y.copy()
for k in idx_2:
   y2.remove(y[k])
ymax2 = max(y2)
idx_3 = [i for i, j in enumerate(y) if j == ymax2]
for i in idx_3:
     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 8)
     print(idx[i])
     
y3 = y2.copy()
for k in idx_3:
   y3.remove(y[k])
ymax3 = max(y3)
idx_4 = [i for i, j in enumerate(y) if j == ymax3]
for i in idx_4:
     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 8)
     print(idx[i])

y4 = y3.copy()
for k in idx_4:
   y4.remove(y[k])
ymax4 = max(y4)
idx_5 = [i for i, j in enumerate(y) if j == ymax4]
for i in idx_5:
     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 8) 
     print(idx[i])
     
y5 = y4.copy()
for k in idx_5:
   y5.remove(y[k])
ymax5 = max(y5)
idx_6 = [i for i, j in enumerate(y) if j == ymax5]
for i in idx_6:
     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 8)
     print(idx[i])
     
y6 = y5.copy()
for k in idx_6:
   y6.remove(y[k])
ymax6 = max(y6)
idx_7 = [i for i, j in enumerate(y) if j == ymax6]
for i in idx_7:
     plt.gca().text(idx[i], y[i], idx[i],color = 'b',fontsize = 8)
     print(idx[i])

imp_map = np.zeros((251,500))
segmentation_fn = SegmentationAlgorithm('slic',n_segments=5000, compactness=1000, max_iter=5, sigma=0.8)
segments = segmentation_fn(imp_map)
for i in idx:   
  imp_map[segments == i]=count[i]

map_3 = np.power(imp_map,3)
plt.figure()
plt.imshow(map_3)
plt.colorbar()

#savemat('sq600_50x_'+region+'_'+sign+'_map_sd'+str(sd)+'_'+str(num_s)+'.mat',{'imp_map': imp_map})