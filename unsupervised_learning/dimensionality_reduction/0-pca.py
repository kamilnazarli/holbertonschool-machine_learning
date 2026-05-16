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
    n = X.shape[0]
    X_centered = X - np.mean(X, axis=0)

    #  covariance matrix
    C = np.dot(X_centered.T, X_centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    sorted_vecs = eigenvectors[np.argsort(eigenvalues)[::-1]]
    total_variance = np.sum(eigenvalues)
    variance_ratio = sorted_vecs / total_variance
    variance_cum = np.cumsum(variance_ratio)
    id = np.where(variance_cum == var)[0]
    X_new = np.dot(X_centered, eigenvectors[:id + 1])
    return X_new