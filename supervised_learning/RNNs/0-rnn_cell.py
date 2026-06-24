#!/usr/bin/env python3
'''RNN'''
import numpy as np


class RNNCell:
    '''RNN class'''
    def __init__(self, i, h, o):
        '''
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs
        '''
        self.Wh, self.Wy = (np.random.randn(i, h),
                            np.random.randn(h, o))
        self.bh, self.by = (np.zeros(1, h),
                            np.zeros(1, o))

    def forward(self, h_prev, x_t):
        '''
        - x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        - m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        '''
        h_next = (x_t @ self.Wh + h_prev + self.bh)
        y = self.sigmoid(h_next @ self.Wy + self.by)
        return h_next, y

    @staticmethod
    def sigmoid(x):
        '''Sigmoid activation function'''
        return 1 / (1 + np.exp(-x))
