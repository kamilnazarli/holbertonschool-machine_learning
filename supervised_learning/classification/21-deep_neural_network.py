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

    def forward_prop(self, X):
        '''forward prop'''
        self.__cache.update({"A0": X})
        for layer in range(1, self.__L+1):
            Z_t = (np.matmul(self.__weights[f"W{layer}"],
                             self.__cache[f"A{layer-1}"])
                   + self.__weights[f"b{layer}"])
            A_t = self.sigmoid(Z_t)
            self.__cache.update({f"A{layer}": A_t})
        return A_t, self.__cache

    def cost(self, Y, A):
        '''cost function'''
        cost_f = - (Y * np.log(A) + (1 - Y) *
                    np.log(1.0000001 - A))
        return np.mean(cost_f)

    def evaluate(self, X, Y):
        '''evaluation'''
        A = self.forward_prop(X)[0]
        pred = np.where(A >= 0.5, 1, 0)
        return pred, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''gradient descent'''
        m = Y.shape[1]
        dZ = {}
        A_last = cache[f"A{self.__L}"]
        dZ[self.__L] = A_last - Y
        weights_copy = self.__weights.copy()
        for layer in range(self.__L, 0, -1):
            A_prev = cache[f"A{layer-1}"] if layer > 1 else cache["A0"]
            # dZ = cache[f"A{layer}"] - Y
            dW = (1 / m) * np.matmul(dZ[layer], A_prev.T)
            db = (1 / m) * np.sum(dZ[layer], axis=1, keepdims=True)

            self.__weights[f"W{layer}"] -= alpha * dW
            self.__weights[f"b{layer}"] -= alpha * db

            if layer > 1:
                dZ[layer-1] = np.dot(weights_copy[f"W{layer}"].T, dZ[layer])
                dZ[layer-1] = (dZ[layer-1] * cache[f"A{layer-1}"] *
                               (1 - cache[f"A{layer-1}"]))

    @staticmethod
    def sigmoid(Z):
        '''sigmoid'''
        return 1 / (1 + np.e ** (-Z))
