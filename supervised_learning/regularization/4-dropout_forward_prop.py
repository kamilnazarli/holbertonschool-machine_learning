#!/usr/bin/env python3
'''module documented'''
import numpy as np


def softmax(x):
    '''softmax implementation'''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def dropout_forward_prop(X, weights, L, keep_prob):
    '''forward prop with dropout'''
    outputs = {"A0": X}
    for i in range(1, L + 1):
        W = weights[f"W{i}"]
        bias = weights[f"b{i}"]
        Z = np.matmul(W, outputs[f"A{i - 1}"]) + bias
        if i != L:
            A = np.tanh(Z)
            d = (np.random.randn(A.shape[0], A.shape[1]) < keep_prob)
            A = A * d
            A /= keep_prob
            outputs[f"D{i}"] = d
        else:
            A = softmax(Z)

        outputs[f"A{i}"] = A
    return outputs
