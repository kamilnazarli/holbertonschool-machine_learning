#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    '''Adam in keras'''
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
