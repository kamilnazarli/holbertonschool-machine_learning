#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def save_model(network, filename):
    '''saver'''
    network.save(filename)
    return None


def load_model(filename):
    '''loader'''
    return K.saving.load_model(filename)
