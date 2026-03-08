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
        if layers.any() <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(1, len(layers)):
            self.W = (np.random.randn(layers[l], layers[l-1]) *
                      np.sqrt(2/layers[l-1]))
            self.b = np.zeros((1, 5))
            self.weights.update({f"W{l}": self.W})
            self.weights.update({f"b{l}": self.b})
            # W1 : W  randn(3, 5)
