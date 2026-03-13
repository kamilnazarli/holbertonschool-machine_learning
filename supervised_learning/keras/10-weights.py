#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    '''saver'''
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    '''loader'''
    network.load_weights(filename)
    return None
