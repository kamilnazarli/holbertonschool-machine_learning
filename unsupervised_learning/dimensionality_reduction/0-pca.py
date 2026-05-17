#!/usr/bin/env python3
'''dimensionality reduction'''
import numpy as np


def pca(X, var=0.95):
    '''
    - X is a numpy.ndarray of shape (n, d) where:
    - n is the number of data points
    - d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
    - var is the fraction of the variance that the
    PCA transformation should maintain
    '''
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered)

    variance = S ** 2
    variance_ratio = variance / np.sum(variance)
    variance_cum = np.cumsum(variance_ratio)
    id =np.argmax(variance_cum >= var) + 1
    Wk = Vt.T[:, :id]
    return Wk