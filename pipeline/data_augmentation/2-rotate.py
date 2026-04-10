#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def rotate_image(image):
    '''
    image is a 3D tf.Tensor
    containing the image to rotate
    '''
    return tf.image.rot90(image)
