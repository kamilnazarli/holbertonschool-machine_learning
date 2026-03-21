#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    '''creating a layer of nn using dropout'''
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=(tf.keras.initializers.VarianceScaling
                            (scale=2.0, mode=("fan_avg")))
                            )(prev)

    dropout = tf.keras.layers.Dropout(rate=1-keep_prob)(
        layer,
        training=training
    )
    return dropout
