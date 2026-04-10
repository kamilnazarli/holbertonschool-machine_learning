#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def change_contrast(image, lower, upper):
    '''
    image: A 3D tf.Tensor representing the
    input image to adjust the contrast.
    lower: A float representing the lower
    bound of the random contrast factor range.
    upper: A float representing the upper
    bound of the random contrast factor range.
    '''
    return tf.image.random_contrast(image,
                                    lower,
                                    upper)
