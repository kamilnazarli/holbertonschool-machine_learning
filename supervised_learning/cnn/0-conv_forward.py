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
    m = A_prev.shape[0]
    if padding == "same":
        ph = int(((h_prev-1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev-1) * sw + kw - w_prev) / 2) + 1
    else:
        ph, pw = 0, 0
    output_h = int(1 + (A_prev.shape[1] + 2 * ph - kh) / sh)
    output_w = int(1 + (A_prev.shape[2] + 2 * pw - kw) / sw)
    output_c = W.shape[3]
    output = np.zeros((m, output_h, output_w, output_c))
    for row in range(output_h):
        for col in range(output_w):
            patch = A_prev[:, row * sh: row * sh + kh,
                           col * sw: col*sw + kw]
            output[:, row, col, :] = np.sum(W * patch, axis=(1, 2)) + b
    
    return activation(output)