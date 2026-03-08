#!/usr/bin/env python3
'''module documented'''
import numpy as np


def one_hot_encode(Y, classes):
    res = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        temp = np.zeros((1, classes))
        temp[0, i] = 1
        res[i] = temp
    return res

