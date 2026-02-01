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
    """
    function3 documented
    """
    if not check(matrix):
        raise TypeError("matrix must be a list of lists")
    
    if not check_shape(matrix):
        raise ValueError("matrix must be square matrix")
    
    n = len(matrix)
    A = [row[:] for row in matrix]
    prev = 1
    sign = 1

    if A == [[]]:
        return 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    A[k], A[r] = A[r], A[k]
                    sign *= -1
                    break
            else:
                return 0 
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[k][k] * A[i][j] - A[i][k] * A[k][j]) // prev
            A[i][k] = 0

        prev = A[k][k]
    return sign * A[n - 1][n - 1]
