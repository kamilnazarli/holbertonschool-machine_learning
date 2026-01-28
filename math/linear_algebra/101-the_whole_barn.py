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


def add_matrices(mat1, mat2):
    '''function2 documented'''
    if len(mat1) == 0 and len(mat2) == 0:
        return []
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None
    result = []
    if len(matrix_shape(mat1)) == 1:
        for i in range(len(mat1)):
            result.append(mat1[i]+mat2[i])
    else:
        for i in range(len(mat1)):
            result.append(add_matrices(mat1[i], mat2[i]))
    return result
