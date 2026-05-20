#!/usr/bin/env python3
'''clustering'''
import numpy as np


def kmeans(X, k, iterations=1000):
    '''- X is a numpy.ndarray of shape (n, d) containing the dataset
       - n is the number of data points
       - d is the number of dimensions for each data point
       - k is a positive integer containing the number of clusters
       - iterations is a positive integer containing the maximum
       number of iterations that should be performed'''
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        return None, None
    n, d = X.shape
    if not (isinstance(k, int) and k <= n and k > 0):
        return None, None
    if not (isinstance(iterations, int) and iterations > 0):
        return None, None
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    cluster_centroids = np.random.uniform(low=min_val,
                                          high=max_val,
                                          size=(k, d))
    clss = np.zeros(n,)
    for i in range(iterations):
        # (n, k)
        distances = np.sqrt(
            np.sum((X[:, np.newaxis] - cluster_centroids) ** 2, axis=(2)))
        clss = np.argmin(distances, axis=1)
        old_c = cluster_centroids.copy()
        for j in range(k):
            if len(X[clss == j]):
                cluster_centroids[j] = (
                    np.mean(X[np.where(clss == j)], axis=0))
            else:
                cluster_centroids[j] = (
                    np.random.uniform(low=min_val,
                                      high=max_val,
                                      size=(d,)))
        if np.all(cluster_centroids == old_c):
            return cluster_centroids, clss
    distances = np.sqrt(
        np.sum((X[:, np.newaxis] - cluster_centroids) ** 2, axis=(2)))
    clss = np.argmin(distances, axis=1)
    return cluster_centroids, clss

