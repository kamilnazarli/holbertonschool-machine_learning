#!/usr/bin/env python3
'''module documented'''
import numpy as np


def one_hot_decode(one_hot):
    '''one hot decoding'''
    res = one_hot.T
    return np.argmax(res, axis=1)
