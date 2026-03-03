#!/usr/bin/env python3
'''module documented'''
import numpy as np


class Neuron:
    '''class documented'''

    def __init__(self, nx):
        '''init documented'''
        if not (isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''getter for W'''
        return self.__W

    @property
    def b(self):
        '''getter for b'''
        return self.__b

    @property
    def A(self):
        '''getter for A'''
        return self.__A
