#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''batch normalization'''
    gamma = [1]
    beta = [0]
    initializer = tf.keras.initializers.VarianceScaling(
        mode="fan_avg"
        )
    Z = tf.keras.layers.Dense(n,
                              kernel_initializer=initializer
                              )(prev)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-7,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones")
    Z_norm = bn(Z)
    activated = activation(Z_norm)
    return activated
