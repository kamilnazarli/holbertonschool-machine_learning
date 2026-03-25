#!/usr/bin/env python3
'''module documented'''
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''same convolution'''
    h, kh = images.shape[1], kernel.shape[0]
    w, kw = images.shape[2], kernel.shape[1]
    if kh == 1 and kw == 1:
        oh, ow = (h - kh + 1), (w - kw + 1)
    else:
        if kh % 2 != 0 and kw % 2 != 0:
            ph, pw = (kh - 1) // 2, (kw - 1) // 2
        else:
            ph, pw = (int(np.ceil((kh - 1) / 2)),
                      int(np.ceil((kw - 1) / 2)))
        oh, ow = (h + 2 * ph - kh + 1), (w + 2 * pw - kw + 1)
        images = np.pad(images,
                        ((0, 0), (ph, ph), (pw, pw)),
                        constant_values=(0))
    output = np.zeros(shape=(images.shape[0], oh, ow))
    for row in range(oh):
        for col in range(ow):
            patch = images[:, row:row+kh, col:col+kw]
            output[:, row, col] = np.sum(patch * kernel, axis=(1, 2))
    return output
