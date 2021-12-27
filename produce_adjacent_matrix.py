
import numpy as np
from math import sqrt
import scipy.io as sio

# 用下面函数可加速算法运行
def corr_x_y(x: np.ndarray, y: np.ndarray):
    """
    Calculate the correlation coefficient between matrix x and y.
    where x in n \times k, y in m \times k
    :param x: matrix, np.ndarray, shape (n, k)
    :param y: matrix, np.ndarray, shape (m, k)
    :return: correlation coefficient, np.ndarray, shape (n, m)
    """
    assert x.shape[1] == y.shape[1], "Different shape!"
    x = x - np.mean(x, axis=1).reshape((-1, 1))
    y = y - np.mean(y, axis=1).reshape((-1, 1))
    lxy = np.dot(x, y.T)
    lxx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    lyy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    std_x_y = np.dot(np.sqrt(lxx), np.sqrt(lyy))
    corr = lxy / std_x_y
    return corr


def cos_similarity(x: np.ndarray, y: np.ndarray):
    """
    Calculate the cos similarity between matrix x and y.
    where x in n \times k, y in m \times k
    :param x: matrix, np.ndarray, shape (n, k)
    :param y: matrix, np.ndarray, shape (m, k)
    :return: cos similarity, np.ndarray, shape (n, m)
    """
    assert x.shape[1] == y.shape[1], "Different shape!"
    xy = np.dot(x, y.T)
    module_x = np.sqrt(np.diag(np.dot(x, x.T))).reshape((-1, 1))
    module_y = np.sqrt(np.diag(np.dot(y, y.T))).reshape((1, -1))
    module_x_y = np.dot(module_x, module_y)
    simi = xy / module_x_y
    return simi


def euclidean_distance(x: np.ndarray, y: np.ndarray):
    """
    Calculate the euclidean distance between matrix x and y.
    where x in n \times k, y in m \times k
    :param x: matrix, np.ndarray, shape (n, k)
    :param y: matrix, np.ndarray, shape (m, k)
    :return: euclidean distance, np.ndarray, shape (n, m)
    """
    assert x.shape[1] == y.shape[1], "Different shape!"
    xy = np.dot(x, y.T)
    xx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    yy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    dist = xx + yy - 2*xy
    dist = np.sqrt(dist)
    return dist


if __name__ == '__main__':
    Data = sio.loadmat('GBM.mat')
    # np.ndarray shape: 213, 12042
    data = Data['GBM_Gene_Expression'].T
    # np.ndarray shape: 213, 1
    targets = Data['GBM_clinicalMatrix']
    # np.ndarray shape: 213, 1
    indexes = Data['GBM_indexes']

    # Data = sio.loadmat('BRCA.mat')
    # data = Data['data'].T
    # targets = Data['targets']
    # indexes = Data['BRCA_indexes']
    # Data = sio.loadmat('LUNG.mat')
    # data = Data['LUNG_Gene_Expression'].T
    # targets = Data['LUNG_clinicalMatrix']
    # indexes = Data['LUNG_indexes']
    z1 =abs(corr_x_y(x=data, y=data))
    normalize_corr = np.where(z1 > 0.8, 1, 0)
    # save data
    f_name = './data/'
    # name1 = f_name + 'BRCA_matrix.mat'
    name1= f_name + 'GBM_matrix.mat'
    sio.savemat(name1, {'normalize_corr':normalize_corr})