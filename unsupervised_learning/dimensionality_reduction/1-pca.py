#!/usr/bin/env python3
'''dimensionality reduction'''
import numpy as np


def pca(X, ndim):
    '''
    - X is a numpy.ndarray of shape (n, d) where:
    - n is the number of data points
    - d is the number of dimensions in each point
    - ndim is the new dimensionality of the transformed X
    '''
    n, d = X.shape[0], X.shape[1]
    X_centered = X - np.mean(X, axis=0)
    #  (n, n) (n, d) (d, d)
    U, S, Vt = np.linalg.svd(X_centered)

    # r = np.argmax(variance_cum >= var) + 1
    Wk = Vt.T
    X_transformed = np.resize(Wk, (d, ndim))
    return X_transformed
