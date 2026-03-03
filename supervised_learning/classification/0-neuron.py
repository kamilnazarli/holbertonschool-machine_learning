#!/usr/bin/env python3
'''module documented'''
import numpy as np


class Neuron:
    '''class documented'''
    def __init__(self, nx):
        if not(isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        w = np.random.normal(loc=1, scale=3, size=10)
        b = 0
        A = 0
