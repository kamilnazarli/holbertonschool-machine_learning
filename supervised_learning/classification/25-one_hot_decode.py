#!/usr/bin/env python3
'''module documented'''
import numpy as np


def one_hot_decode(one_hot):
    '''one hot decoding'''
    temp = np.zeros((one_hot.shape[1],), dtype=int)
    for i in range(one_hot.shape[0]):
        temp[i] = np.argmax(one_hot[i])
    return temp
