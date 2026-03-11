#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''function documented'''
    layer = K.layers.CategoryEncoding(
        num_tokens=classes,
        output_mode="one_hot"
    )
    return layer(labels)
