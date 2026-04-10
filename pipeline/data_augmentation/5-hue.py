#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def change_hue(image, delta):
    '''
    image is a 3D tf.Tensor
    containing the image to change
    delta is the amount the hue should change
    '''
    img = tf.image.random_hue(image,
                               delta)
    return img
