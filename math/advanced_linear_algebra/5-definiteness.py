#/usr/bin/env python3
import numpy as np
'''module documented'''


def matrix_transpose(matrix):
    '''transpose function'''
    new_matrix = []
    for col in range(len(matrix[0])):
        temp = []
        for row in range(len(matrix)):
            temp.append(matrix[row][col])
        new_matrix.append(temp)
    return new_matrix


def definiteness(matrix):
    '''function documented'''
    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    x = len(matrix)
    if any(len(row) != x for row in matrix):
        return None
    if matrix_transpose(matrix) != matrix:
        return None
    eigen_values = np.linalg.eigvalsh(matrix)
    if all(eigen_values > 0):
        return 'Positive definite'
    elif all(eigen_values >= 0):
        return 'Positive semi-definite'
    elif all(eigen_values < 0):
        return 'Negative definite'
    elif all(eigen_values <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'