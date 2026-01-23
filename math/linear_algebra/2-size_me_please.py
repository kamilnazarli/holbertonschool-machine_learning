#!/usr/bin/env python3
'''module documented'''


def matrix_shape(matrix):
    '''function documented'''
    shape = []
    current = matrix
    while True:
        if type(current) is list:
            shape.append(len(current))
            current = current[0]
        else:
            break
    return shape
