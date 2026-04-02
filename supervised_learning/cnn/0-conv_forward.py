#!/usr/bin/env python3
'''module documented'''
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b is a numpy.ndarray of shape (1, 1, 1, c_new)
        activation is an activation function
        stride is a tuple of (sh, sw)
        padding is a string that is either same or valid,
        indicating the type of padding used
    '''
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    sh, sw = stride
    kh, kw = W.shape[0], W.shape[1]
    if padding == "same":
        ph = int(((h_prev-1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev-1) * sw + kw - w_prev) / 2) + 1
    else:
        ph, pw = 0
    output_h = 1 + (A_prev.shape[1] + 2 * ph - W.shape[1]) / sh
    output_w = 1 + (A_prev.shape[2] + 2 * pw - W.shape[2]) / sw
    output_c = W.shape[3]
    output = np.zeros((output_h, output_w, output_c))
    for row in range(output_h):
        for col in range(output_w):
            output[row, col, :] = np.sum(W[row, col, :, :] * A_prev[:, row, col, :], axis=(1, 2)) + b
    
    return activation(output)