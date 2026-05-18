#!/usr/bin/env python3
'''clustering'''
import numpy as np


def initialize(X, k):
    '''- X is a numpy.ndarray of shape (n, d) containing the dataset
       that will be used for K-means clustering
       - n is the number of data points
       - d is the number of dimensions for each data point
       - k is a positive integer containing the number of clusters'''
    d = X.shape[1]
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    cluster_centroids = np.random.uniform(low=min_val,
                                          high=max_val,
                                          size=(k, d))
    return cluster_centroids
