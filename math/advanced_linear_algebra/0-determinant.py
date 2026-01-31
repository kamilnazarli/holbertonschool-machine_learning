#!/usr/bin/env python3
'''module documented'''


def determinant(matrix):
    '''function documented''' 
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
    return A
