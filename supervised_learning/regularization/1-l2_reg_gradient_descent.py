#!/usr/bin/env python3
'''module documented'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lamtha, L):
    '''l2 gradient descent'''
    m = Y.shape[1]  # number of data points
    k = Y.shape[0]  # classes
    dz = cache[f"A{L}"]
    for i in range(L, 0, -1):
        W = weights[f"W{i}"]
        A_prev = cache[f"A{i - 1}"]

        dW = (1 / m) * (np.matmul(dZ, A_prev.T) + lamtha * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        bias = weights[f"b{i}"]
        if i > 1:
            A_prev = cache[f"A{i - 1}"]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev**2)

        weights[f"W{i}"] = weights[f"W{i}"] - alpha * dW
        weights[f"b{i}"] = weights[f"b{i}"] - alpha * db
