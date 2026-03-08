#!/usr/bin/env python3
'''module documented'''
import numpy as np


def one_hot_encode(Y, classes):
    res = np.zeros((classes, len(Y)))
    for i in range(classes):
        temp = np.zeros((1, len(Y)))
        temp[0, Y[i]] = 1
        res[i] = temp
    return res.T

