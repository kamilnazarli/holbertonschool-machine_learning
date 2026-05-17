#!/usr/bin/env python3
'''dimensionality reduction'''
import numpy as np


def pca(X, var=0.95):
    '''
    - X is a numpy.ndarray of shape (n, d) where:
    - n is the number of data points
    - d is the number of dimensions in each point
    - ndim is the new dimensionality of the transformed X
    '''
    X_centered = X - np.mean(X, axis=0)
    #  (n, n) (n, d) (d, d)
    U, S, Vt = np.linalg.svd(X_centered)

    variance = S
    variance_ratio = variance / np.sum(variance)
    variance_cum = np.cumsum(variance_ratio)
    r = np.argmax(variance_cum >= var) + 1
    Wk = Vt.T[:, :r]
    return np.dot(X_centered, Wk)
