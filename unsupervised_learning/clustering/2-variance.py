#!/usr/bin/env python3
'''clustering'''
import numpy as np


def variance(X, C):
    '''
    - X is a numpy.ndarray of shape (n, d) containing the data set
    - C is a numpy.ndarray of shape (k, d)
    containing the centroid means for each cluster
    '''
    n = X.shape[0]
    var = np.sum((X[:, np.newaxis] - C) ** 2, axis=(2, 0)) / n
    return var

