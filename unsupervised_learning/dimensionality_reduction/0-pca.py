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
    sorted_ids = np.argsort(eigenvalues)[::-1]
    sorted_vals = eigenvalues[sorted_ids]
    sorted_vecs = eigenvectors[:, sorted_ids]
    total_variance = np.sum(eigenvalues)
    variance_ratio = sorted_vals / total_variance
    variance_cum = np.cumsum(variance_ratio)
    id =np.argmax(variance_cum >= var) + 1
    Wk = sorted_vecs[:, : id]
    Wk_ = sorted_vecs[:, : id - 1]
    X_new = np.dot(X_centered, Wk)
    X_new_ = np.dot(X_centered, Wk_)
    return X_new, X_new_