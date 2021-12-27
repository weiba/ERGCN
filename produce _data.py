
import numpy as np
import time
import pandas as pd
import os
import sys
import scipy.io as sio
from math import sqrt
import scipy.sparse as sp
from sklearn.model_selection import KFold


A=sio.loadmat('./data/GBM_matrix.mat')
# A=sio.loadmat('./data/LUNG_matrix.mat')
# A=sio.loadmat('./data/BRCA_matrix.mat')
A=A['normalize_corr']

edges=sp.coo_matrix(A)

file_write_obj = open("./data/edges_gbm.csv", 'w+')
for id in np.arange(edges.data.shape[0]):
    file_write_obj.writelines(np.str(edges.row[id]))
    file_write_obj.write(',')
    file_write_obj.writelines(np.str(edges.col[id]))
    file_write_obj.write('\n')
file_write_obj.close()

