#!/usr/bin/env python3
'''module documented'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''function documented'''
    result = mat1[::]
    if axis == 0:
        result += mat2
    else:
        for i in range(len(mat2)):
            result[i] += mat2[i]
    return result
