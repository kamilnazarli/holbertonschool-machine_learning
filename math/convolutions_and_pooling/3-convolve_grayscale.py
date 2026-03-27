#!/usr/bin/env python3
'''module documented'''
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    '''same convolution'''
    h, kh = images.shape[1], kernel.shape[0]
    w, kw = images.shape[2], kernel.shape[1]
    sh, sw = stride
    if padding == "same":
        oh, ow = int(np.ceil(h / sh)), int(np.ceil(w / sw))
        ph = int(np.ceil(((oh - 1) * sh + kh - h)))
        pw = int(np.ceil(((ow - 1) * sw + kw - w)))
        pt = int(np.floor(ph / 2))
        pl = int(np.floor(pw / 2))
        pb, pr = ph - pt, pw - pl
        images = np.pad(images,
                        ((0, 0), (pt, pb), (pl, pr)),
                        constant_values=(0))
    elif padding == "valid":
        oh, ow = (int(np.ceil((h - kh + 1) / sh)),
                  int(np.ceil((w - kw + 1) / sw)))

    else:
        ph, pw = padding
        oh, ow = (int(np.floor((h - kh + 2 * ph) / sh)) + 1,
              (int(np.floor((w - kw + 2 * pw) / sw)) + 1))

        images = np.pad(images,
                        ((0, 0), (ph, ph), (pw, pw)),
                        constant_values=(0))
    output = np.zeros(shape=(images.shape[0], oh, ow))
    for row in range(oh):
        for col in range(ow):
            patch = images[:, row * sh : row * sh + kh,
                           col * sw : col * sw + kw]
            output[:, row, col] = np.sum(patch * kernel, axis=(1, 2))
    return output
