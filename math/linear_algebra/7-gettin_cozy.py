#!/usr/bin/env python3
'''module documented'''


def matrix_shape(matrix):
    '''function1 documented'''
    shape = []
    current = matrix
    while True:
        if type(current) is list and len(current) > 0:
            shape.append(len(current))
            current = current[0]
        else:
            break
    return shape

def cat_matrices2D(mat1, mat2, axis=0):
    '''function2 documented'''
    result = mat1[::]
    if axis == 0:
        result += mat2
    else:
        for i in range(len(mat2)):
            result[i] += mat2[i]
    return result
