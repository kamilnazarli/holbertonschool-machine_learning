#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def flip_image(image):
    '''image is a 3D tf.Tensor
       containing the image to flip'''
    res = tf.image.flip_left_right(image)
    return res
