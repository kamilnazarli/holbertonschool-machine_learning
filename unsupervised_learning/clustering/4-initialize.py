#!/usr/bin/env python3
'''clustering'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    '''
    - X is a numpy.ndarray of shape (n, d) containing the data set
    - k is a positive integer containing the number of clusters
    - pi is a numpy.ndarray of shape (k,) containing the priors
    for each cluster, initialized evenly
    - m is a numpy.ndarray of shape (k, d) containing the
    centroid means for each cluster, initialized with K-means
    - S is a numpy.ndarray of shape (k, d, d) containing the
    covariance matrices for each cluster, initialized as identity matrices
    '''
    n, d = X.shape
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        return None, None, None
    if not (isinstance(k, int) and k <= n and k > 0):
        return None, None, None
    m, clss = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))
    pi = np.full((k,), 1 / k)
    return pi, m, S
