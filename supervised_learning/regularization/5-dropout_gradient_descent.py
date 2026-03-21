#!/usr/bin/env python3
'''module documented'''
import numpy as np


def softmax(x):
    '''softmax implementation'''
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''gradient descent with dropout'''
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y
    for i in range(L, 0, -1):
        W = weights[f"W{i}"]
        A_prev = cache[f"A{i - 1}"]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev**2)  # tanh
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
