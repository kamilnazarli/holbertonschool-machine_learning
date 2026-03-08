#!/usr/bin/env python3
'''module documented'''
import numpy as np


def one_hot_decode(one_hot):
    '''one hot decoding'''
    res = one_hot.T
    temp = np.zeros((one_hot.shape[1],), dtype=int)
    for i in range(one_hot.shape[1]):
        temp[i] = np.argmax(res[i])
    return temp
