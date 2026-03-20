#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''creating nn layer using L2'''
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        kernel_regularizer=tf.keras.regularizers.l2(lambtha),
        bias_initializer=tf.keras.initializers.Zeros()
    )(prev)
    return layer
