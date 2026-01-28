#!/usr/bin/env python3
'''module documented'''


def cat_matrices(mat1, mat2, axis=0):
    '''function documented'''
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
