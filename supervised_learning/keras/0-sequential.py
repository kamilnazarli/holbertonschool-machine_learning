#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''function documented'''
    model = K.models.Sequential()
    model.add(K.layers.Dense(layers,
                             input_shape=(nx,),
                             activation=activations))
    return model
