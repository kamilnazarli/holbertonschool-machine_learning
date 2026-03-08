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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        # L = 3     1, 2
        for layer in range(self.__L):
            if layers[layer] <= 0 or not (isinstance(layer, int)):
                raise TypeError("layers must be a list of positive integers")
            if layer == 0:
                prev = nx
            else:
                prev = layers[layer-1]
            W = (np.random.randn(layers[layer], prev) *
                 np.sqrt(2 / prev))
            b = np.zeros((layers[layer], 1))
            self.__weights.update({f"W{layer+1}": W})
            self.__weights.update({f"b{layer+1}": b})
            # W1 : W  randn(3, 5)
            # b1 : b zeros()

    @property
    def L(self):
        '''getter'''
        return self.__L
    @property
    def cache(self):
        '''getter'''
        return self.__cache
    @property
    def weights(self):
        '''getter'''
        return self.__weights
