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
        dist = (X1 - X2.T) ** 2
        # dist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
        # np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T))
        exponential = np.exp(-dist / (2 * self.l ** 2))
        k_rbf = (self.sigma_f ** 2) * exponential
        return k_rbf

    def predict(self, X_s):
        '''
        - X_s is a numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated
        '''
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        mu = (K_s.T @ K_inv @ self.Y).reshape(-1)
        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov)
        return mu, sigma

    def update(self, X_new, Y_new):
        '''
        - X_new is a numpy.ndarray of shape (1,) that
        represents the new sample point
        - Y_new is a numpy.ndarray of shape (1,) that
        represents the new sample function value
        '''
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(X_new, X_new)
