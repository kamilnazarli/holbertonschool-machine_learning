#!/usr/bin/env python3
'''hyperparameter tuning'''
import numpy as np


class GaussianProcess:
    ''' noiseless 1D Gaussian process'''
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        '''
        - X_init is a numpy.ndarray of shape (t, 1) representing
        the inputs already sampled with the black-box function
        - Y_init is a numpy.ndarray of shape (t, 1) representing
        the outputs of the black-box function for each input in X_init
        - t is the number of initial samples
        - l is the length parameter for the kernel
        - sigma_f is the standard deviation given to the output
        of the black-box function'''
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        '''
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        '''
        dist = - (X1 - X2.T) ** 2
        # dist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
                # np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T))
        exponential = np.exp(dist / (2 * self.l ** 2))
        k_rbf = (self.sigma_f ** 2) * exponential
        return k_rbf
