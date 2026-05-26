#!/usr/bin/env python3
'''clustering'''
import numpy as np


def pdf(X, m, S):
    '''
    - X is a numpy.ndarray of shape (n, d) containing
    the data points whose PDF should be evaluated
    - m is a numpy.ndarray of shape (d,) containing
    the mean of the distribution
    - S is a numpy.ndarray of shape (d, d) containing
    the covariance of the distribution
    '''
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        return None
    if not (isinstance(m, np.ndarray) and m.ndim == 2):
        return None
    if not (isinstance(S, np.ndarray) and S.ndim == 2):
        return None

    if m.shape != (d,):
        return None

    if S.shape != (d, d):
        return None
    
    n, d = X.shape
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    diff = X - m
    norm_constant = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    exponent = np.exp(-0.5 * np.sum((diff @ inv) * diff, axis=1))
    P = norm_constant * exponent
    P = np.maximum(P, 1e-300)
    return P
