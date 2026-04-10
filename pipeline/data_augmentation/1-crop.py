#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def crop_image(image, size):
    '''image is a 3D tf.Tensor
       containing the image to crop
       size is a tuple containing
       the size of the crop'''
    res = tf.image.random_crop(image,
                               size=size)
    return res
