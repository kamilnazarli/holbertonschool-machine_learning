#!/usr/bin/env python3
'''neural style from scratch'''
import numpy as np
import tensorflow as tf


class NST:
    '''class NST'''
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        '''
        style_image - the image used as a style reference,
        stored as a numpy.ndarray
        content_image - the image used as a content reference,
        stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        '''
        if not (isinstance(style_image, np.ndarray) and
                style_image.shape[2]==3 and
                len(style_image.shape)==3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not (isinstance(content_image, np.ndarray) and
                content_image.shape[2]==3 and
                len(content_image.shape)==3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if alpha < 0 or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a non-negative number")
        if beta < 0 or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        '''image - a numpy.ndarray of shape (h, w, 3)
           containing the image to be scaled
        '''
        if not (isinstance(image, np.ndarray) and
                image.shape[2]==3 and
                len(image.shape)==3):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        h, w = image.shape[0], image.shape[1]
        scale = 512 / max(h, w)
        image = np.expand_dims(image, axis=0)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        h_new, w_new = int(h * scale), int(w * scale)
        scaled_image = tf.image.resize(image_tensor,
                                       size=(h_new, w_new),
                                       method='bicubic')
        scaled_image /= 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0, 1)
        return scaled_image
