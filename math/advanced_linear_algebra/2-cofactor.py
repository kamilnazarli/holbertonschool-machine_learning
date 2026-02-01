#!/usr/bin/env python3
"""
Determinant of a matrix
"""


def determinant(mat):
    """
    Calculation of determinant of a matrix
    """
    x = len(mat)

    if x == 1:
        return mat[0][0]
    if x == 2:
        return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]

    return sum(
        (-1) ** k * mat[0][k] *
        determinant([row[:k] + row[k + 1:] for row in mat[1:]])
        for k in range(x)
    )


def minor(matrix):
    '''Minor function'''
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    x = len(matrix)
    if any(len(row) != x for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if x == 1:
        return [[1]]
    minor = [[0 for _ in range(x)] for _ in range(x)]
    for i in range(x):
        for j in range(x):
            sub_matrix = [
                [matrix[m][n] for n in range(x) if n != j]
                for m in range(x) if m != i]
            minor[i][j] = determinant(sub_matrix)
    return minor


def cofactor(matrix):
    '''cofactor function'''
    minor_matrix = minor(matrix)
    n = len(minor_matrix)
    for i in range(n):
        for j in range(n):
            minor_matrix[i][j] = (-1) ** (i + j) * minor_matrix[i][j]
    return minor_matrix

