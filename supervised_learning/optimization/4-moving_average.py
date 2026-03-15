#!/usr/bin/env python3
'''module documented'''
import numpy as np


def moving_average(data, beta):
    """method"""
    ewa = [data[0]]
    for i in range(1, len(data)):
        ewa.append(beta * ewa[i - 1] + (1 - beta) * data[i])
    return ewa
