#!/usr/bin/env python3
'''clustering'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''
    - X is a numpy.ndarray of shape (n, d) containing
    the data set
    - kmin is a positive integer containing the minimum
    number of clusters to check for (inclusive)
    - kmax is a positive integer containing the maximum
    number of clusters to check for (inclusive)
    - iterations is a positive integer containing the
    maximum number of iterations for K-means
    '''
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        return None, None
    n, d = X.shape
    if not (isinstance(kmin, int) and kmin > 0):
        return None, None
    if not (isinstance(kmax, int) and kmax > 0):
        return None, None
    if kmin > kmax:
        return None, None
    if not (isinstance(iterations, int) and iterations > 0):
        return None, None
    results, variances = [], []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        variances.append(variance(X, C))
    d_vars = [variances[0] - v for v in variances]
    return results, d_vars
