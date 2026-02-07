#!/usr/bin/env python3
'''module documented'''
import numpy as np


def pmf(n, k, p):
        '''prob mass function for binomial'''
        if not (isinstance(k, int)):
            k = int(k)
        if k < 0:
            return 0
        return ((factorial(n) /
                 ((factorial(k) * factorial(n - k))) *
                 (p ** k) * (1 - p) ** (n - k)))

def factorial(n):
        '''factorial documented'''
        fact = 1
        for i in range(1, n + 1):
            fact = fact * i
        return fact

def likelihood(x, n, P):
    '''likelihood documented'''
    if not (n >= 0 and isinstance(n, int)):
        raise ValueError("n must be a positive integer")
    if not (isinstance(x, int) and x >= 0):
        raise ValueError("x must be an integer that is " \
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not (isinstance(P, np.ndarray) and len(P.shape)==1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if any(P < 0) or any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    return pmf(n, x, P)
