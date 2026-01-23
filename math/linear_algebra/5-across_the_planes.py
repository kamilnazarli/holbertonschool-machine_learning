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


def add_matrices2D(mat1, mat2):
    '''function2 documented'''
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    result = []
    for row in range(len(mat1)):
        temp_s = []
        for col in range(len(mat1[row])):
            temp_s.append(mat1[row][col]+mat2[row][col])
        result.append(temp_s)
    return result
