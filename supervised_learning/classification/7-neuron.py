#!/usr/bin/env python3
'''module documented'''
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        '''forward propogation'''
        s = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(s)
        return self.__A

    def cost(self, Y, A):
        '''cost function'''
        x = 1.0000001 - A
        cost_f = -np.mean((Y * np.log(A)) + (1 - Y) * np.log(x))
        return cost_f

    def evaluate(self, X, Y):
        '''evaulation'''
        pred = self.forward_prop(X)
        pred = np.where(pred > 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''gradient descent'''
        m = Y.shape[1]
        dZ = A - Y  # x - (nx, m), (1, m)
        dw = np.dot(dZ, X.T) / m  # (1, nx)
        db = np.sum(dZ) / m  # (1, m)
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose= True, graph=True, step=100):
        '''training'''
        if not (isinstance(iterations, int)):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not (isinstance(alpha, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not (isinstance(step, int)):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
    
        iteration_s, cost_s = [], []
        for i in range(iterations+1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if verbose:
                if i % step == 0:
                    print(f"Cost after {i} iterations: {self.cost(Y, A)}")
                    iteration_s.append(i)
                    cost_s.append(self.cost(Y, A))
        if graph:
            plt.plot(iteration_s, cost_s)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)

    @staticmethod
    def sigmoid(X):
        '''sigmoid function'''
        return 1 / (1 + np.e ** (-X))
