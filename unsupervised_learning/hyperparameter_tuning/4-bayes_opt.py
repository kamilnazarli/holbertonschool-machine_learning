#!/usr/bin/env python3
'''hyperparameter tuning'''
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''
    Bayesian optimization on a noiseles
    1D Gaussian process
    '''
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        '''
        - f is the black-box function to be optimized
        - X_init is a numpy.ndarray of shape (t, 1) representing
        the inputs already sampled with the black-box function
        - Y_init is a numpy.ndarray of shape (t, 1) representing the
        outputs of the black-box function for each input in X_init
        - t is the number of initial samples
        - bounds is a tuple of (min, max) representing the bounds of
        the space in which to look for the optimal point
        - ac_samples is the number of samples that should be analyzed
        during acquisition
        - l is the length parameter for the kernel
        - sigma_f is the standard deviation given to the output of the
        black-box function
        - xsi is the exploration-exploitation factor for acquisition
        - minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False)

        '''
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        '''
        calculates the next best sample location
        '''
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1)
        mu = mu.reshape(-1)
        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi
        EI = np.zeros_like(mu)
        nonzero_sigma = sigma > 0
        Z = np.zeros_like(mu)
        Z[nonzero_sigma] = (
            improvement[nonzero_sigma] / sigma[nonzero_sigma])
        EI[nonzero_sigma] = (improvement[nonzero_sigma] * norm.cdf(Z[nonzero_sigma]) +
                             sigma[nonzero_sigma] * norm.pdf(Z[nonzero_sigma]))
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
