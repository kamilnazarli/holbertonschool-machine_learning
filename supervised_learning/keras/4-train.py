#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    '''funcion documented'''
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size
                          )
    return history
