#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''creating nn layer using L2'''
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode=("fan_avg")
    )
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )(prev)
    return layer
