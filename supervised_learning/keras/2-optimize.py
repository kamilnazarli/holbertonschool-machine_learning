#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''function documented'''
    opt = K.optimizers.Adam(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2
        )
    network.compile(loss="categorical_crossentropy", 
                    optimizer=opt
                    )
    return None
