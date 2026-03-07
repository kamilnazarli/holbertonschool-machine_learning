#!/usr/bin/env python3
'''module documented'''
import numpy as np


class NeuralNetwork:
    '''class documented'''
    def __init__(self, nx, nodes):
        '''init documented'''
        if not (isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not (isinstance(nodes, int)):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''getter for W1'''
        return self.__W1

    @property
    def b1(self):
        '''getter for b1'''
        return self.__b1

    @property
    def A1(self):
        '''getter for A1'''
        return self.__A1

    @property
    def W2(self):
        '''getter for W2'''
        return self.__W2

    @property
    def b2(self):
        '''getter for b2'''
        return self.__b2

    @property
    def A2(self):
        '''getter for A2'''
        return self.__A2

    def forward_prop(self, X):
        '''forward propagation'''
        # X(nx, m), W1(nodes, nx), Z1(nodes, m)
        # W2(1, nodes), output(1, m)
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''cost funcrion'''
        cost_f = -(Y * np.log(A) + (1 - Y)
                   * np.log(1.0000001 - A))
        return np.mean(cost_f)

    def evaluate(self, X, Y):
        '''evaluation'''
        pred = self.forward_prop(X)[1]
        pred = np.where(pred >= 0.5, 1, 0)
        return pred, self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''gradient'''
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    @staticmethod
    def sigmoid(Z):
        '''sigmoid func'''
        return 1 / (1 + np.e ** (-Z))
