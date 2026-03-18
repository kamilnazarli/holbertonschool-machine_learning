#!/usr/bin/env python3
'''module documented'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''l2 regularization'''
    s = 0
    for i in range(1, L + 1):
        s += np.sum(weights[f"W{i}"] ** 2)
    cost += s * lambtha / (2 * m)
    return cost
