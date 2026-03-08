#!/usr/bin/env python3
'''module documented'''
import numpy as np


class DeepNeuralNetwork:
    '''class documented'''
    def __init__(self, nx, layers):
        '''init documented'''
        if not (isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not (isinstance(layers, list)) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        # L = 3     1, 2
        for l in range(self.L):
            if layers[l] <= 0 or not (isinstance(l, int)):
                raise TypeError("layers must be a list of positive integers")
            if l == 0:
                prev = nx
            else:
                prev = layers[l-1]
            W = (np.random.randn(layers[l], prev) *
                      np.sqrt(2 / prev))
            b = np.zeros((layers[l], 1))
            self.weights.update({f"W{l+1}": W})
            self.weights.update({f"b{l+1}": b})
            # W1 : W  randn(3, 5)
            # b1 : b zeros()
