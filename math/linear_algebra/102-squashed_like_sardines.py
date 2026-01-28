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

def cat_mat1D(mat1, mat2):
    '''function1 documented'''
    result = mat1[::]
    for i in mat2:
        result.append(i)
    return result

def cat_matrices(mat1, mat2, axis=0):
    '''function2 documented'''
    if len(matrix_shape(mat1)) == 1 and len(matrix_shape(mat2)) == 1:
        return cat_mat1D(mat1, mat2)
    result = mat1[::]
    if axis == 0:
        result += mat2
        if len(result) > 0 and len(mat2) > 0:
            if len(mat1[0]) != len(mat2[0]):
                return None
    else:
        if len(mat1) != len(mat2):
            return None
        for i in range(min(len(result), len(mat2))):
            result[i] += mat2[i]
    return result
