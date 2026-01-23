#!/usr/bin/env python3
'''module documented'''

def matrix_shape(matrix):
    '''function1 documented'''
    shape = []
    current = matrix
    while True:
        if type(current) is list:
            shape.append(len(current))
            current = current[0]
        else:
            break
    return shape

def add_arrays(arr1, arr2):
    '''function2 documented'''
    shape1 = matrix_shape(arr1)
    shape2 = matrix_shape(arr2)
    if len(arr1) == 0 and len(arr2) == 0:
        return None
    elif shape1 != shape2:
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i]+arr2[i])
    return result
