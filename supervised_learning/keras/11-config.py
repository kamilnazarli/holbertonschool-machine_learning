#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def save_config(network, filename):
    '''saver'''
    json_string = network.to_json()
    with open(filename, "w") as f:
        f.write(json_string)
    return None


def load_config(filename):
    '''loader'''
    with open(filename, "r") as f:
        json_config = f.read()
    return K.models.model_from_json(json_config)
