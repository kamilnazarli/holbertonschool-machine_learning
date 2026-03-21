#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    '''creating a layer of nn using dropout'''
    tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    dropout = tf.keras.layers.Dropout(
        rate=keep_prob,
        trainable=training)
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        callbacks=[dropout])
    return layer(prev)
