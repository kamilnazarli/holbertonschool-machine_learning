#!/usr/bin/env python3
'''module documented'''
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''testing model'''
    return network.evaluate(data, labels, verbose=verbose)
