#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def save_config(network, filename):
    '''saver'''
    network.get_config(filename)
    return None


def load_config(filename):
    '''loader'''
    return K.models.model_from_json(filename)
