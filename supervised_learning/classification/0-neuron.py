#!/usr/bin/env python3
'''module documented'''

class Neuron:
    '''class documented'''
    def __init__(self, nx):
        if not(isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
