#!/usr/bin/env python3
'''module documented'''
import numpy as np


def one_hot_encode(Y, classes):
    """one-hot-encoding"""
    res = np.zeros((len(Y), classes))
    for i in range(len(Y)):
        temp = np.zeros((1, classes))
        temp[0, Y[i]] = 1
        res[i] = temp
    return res
