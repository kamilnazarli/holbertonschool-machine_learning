#!/usr/bin/env python3
'''module documented'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lamtha, L):
    '''l2 gradient descent'''
    m = Y.shape[1]  # number of data points
    k = Y.shape[0]  # classes
    for i in range(1, L + 1):
        W = weights[f"W{i}"]
        A_prev = cache[f"A{i - 1}"]
        bias = weights[f"b{i}"]
        Z = np.matmul(W, A_prev) + bias
        if i != L:
            A = np.tanh(Z)
        else:
            # softmax implementation
            z_shifted = Z - np.max(Z, axis=0, keepdims=True)
            A = (np.exp(z_shifted) /
                 np.sum(np.exp(z_shifted), axis=0, keepdims=True))
        dZ = A - Y
        dW = (1 / m) * (np.matmul(dZ, A_prev) + lamtha * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights[f"W{i}"] = weights[f"W{i}"] - alpha * dW
        weights[f"b{i}"] = weights[f"b{i}"] - alpha * db
