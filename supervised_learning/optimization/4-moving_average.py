#!/usr/bin/env python3
'''module documented'''
import numpy as np


def moving_average(data, beta):
    '''method'''
    mat = [0]
    for i in range(1, len(data)):
        mat.append(beta * mat[i - 1] + (1 - beta) * data[i])
        mat[i] = mat[i] / (1 - beta ** i)
    return mat
