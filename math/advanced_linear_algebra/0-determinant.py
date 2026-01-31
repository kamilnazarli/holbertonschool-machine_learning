#!/usr/bin/env python3
'''module documented'''
def check_shape(matrix):
    '''function1 documented'''
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return True
    return len(matrix)==len(matrix[0])

def check(ls):
    '''function2 documented'''
    if type(ls) is not list:
        return False

    if len(ls) == 0:
        return False

    for row in ls:
        if type(row) is not list:
            return False
    return True

def determinant(matrix):
    '''function3 documented'''
    if not check(matrix):
        raise TypeError("matrix must be a list of lists")
    
    if not check_shape(matrix):
        raise ValueError("matrix must be square matrix")
     
    A = [row[:] for row in matrix]
    rows = len(A)
    cols = len(A[0])
    pivot_row = 0
    for pivot_col in range(cols):
        if pivot_row >= rows:
            break
        if A[pivot_row][pivot_col] == 0:
            for r in range(pivot_row + 1, rows):
                if A[r][pivot_col] != 0:
                    A[pivot_row], A[r] = A[r], A[pivot_row]
                    break
        if A[pivot_row][pivot_col] == 0:
            continue
        for r in range(pivot_row+1, rows):
            factor = A[r][pivot_col] / A[pivot_row][pivot_col]
            for c in range(cols):
                A[r][c] = A[r][c] - factor * A[pivot_row][c]
        pivot_row += 1

    det = 1
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == j:
                det *= A[i][j]
    
    return det
