#!/usr/bin/env python3
'''module documented'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


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
        A_last = cache[f"A{self.__L}"]
        dZ = A_last - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache[f"A{layer-1}"]
            Wl = self.__weights[f"W{layer}"]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{layer}"] = (self.__weights[f"W{layer}"] -
                                           alpha * dW)
            self.__weights[f"b{layer}"] = (self.__weights[f"b{layer}"] -
                                           alpha * db)

            if layer > 1:
                A_prev_layer = cache[f"A{layer-1}"]
                dZ = np.dot(Wl.T, dZ) * (A_prev_layer * (1 - A_prev_layer))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''training'''
        if not (isinstance(iterations, int)):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not (isinstance(alpha, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            cache = self.forward_prop(X)[1]
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        '''upgraded training'''
        if not (isinstance(iterations, int)):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not (isinstance(alpha, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (isinstance(step, int)):
            raise TypeError("step must be an integer")
        if iterations <= 0 and step >= iterations:
            raise ValueError("step must be positive and <= iterations")
        if verbose or graph:
            if not (isinstance(step, int)):
                raise TypeError("iterations must be an integer")
            if step <= 0 or step >= iterations:
                raise TypeError("step must be positive and <= iterations")
        iteration_s, cost_s = [], []
        for i in range(iterations+1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % step == 0:
                cost_f = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost_f}")
                iteration_s.append(i)
                cost_s.append(cost_f)
        if graph:
            plt.plot(iteration_s, cost_s)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        '''saving'''
        obj = DeepNeuralNetwork(3, [4, 5, 6])

        if not (".pkl" in filename):
            filename = f"{filename}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

        if not (os.path.exists(filename)):
            return None

        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data
        
    @staticmethod
    def sigmoid(Z):
        '''sigmoid'''
        return 1 / (1 + np.e ** (-Z))
