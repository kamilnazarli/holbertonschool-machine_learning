#!/usr/bin/env python3
'''RNN implementation'''
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    - X is the data to be used, given
    as a numpy.ndarray of shape (t, m, i)
    - h_0 is the initial hidden state,
    given as a numpy.ndarray of shape (m, h)
    """
    H, Y = [h_0], []
    for x_t in X:
        h_next, y = rnn_cell.forward(h_0, x_t)
        H.append(h_next)
        Y.append(y)
    return H, Y
