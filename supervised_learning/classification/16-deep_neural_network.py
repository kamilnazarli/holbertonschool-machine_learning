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
            raise TypeError("nx must be a positive integer")
        if not (isinstance(layers, list)) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if any(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        # L = 3     1, 2 
        for l in range(1, self.L):
            W = (np.random.randn(layers[l], layers[l-1]) *
                      np.sqrt(2/layers[l-1]))
            b = np.zeros((layers[l], 1))
            self.weights.update({f"W{l}": W})
            self.weights.update({f"b{l}": b})
            # W1 : W  randn(3, 5)
            # b1 : b zeros()
