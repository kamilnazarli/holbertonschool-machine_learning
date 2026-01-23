#!/usr/bin/env python3
def matrix_transpose(matrix):
    new_matrix = []
    for col in range(len(matrix[0])):
        temp = []
        for row in range(len(matrix)):
            temp.append(matrix[row][col])
        new_matrix.append(temp)
    return new_matrix
