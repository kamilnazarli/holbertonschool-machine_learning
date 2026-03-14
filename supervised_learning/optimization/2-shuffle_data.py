#!/usr/bin/env python3
'''module documented'''
import numpy as np


def shuffle_data(X, Y):
    '''function'''
    X_shuffled = np.random.permutation(X)
    y_shuffled = np.random.permutation(Y)
    return X_shuffled, y_shuffled
