#!/usr/bin/env python3
'''module documented'''


def mat_mul(mat1, mat2):
    '''function documented'''
    if len(mat1[0]) != len(mat2):
        return None
    res = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            temp = [] # keeps multiples for sum
            for k in range(len(mat1[0])):
                temp.append(mat1[i][k]*mat2[k][j])
            new_row.append(sum(temp))
        res.append(new_row)
    return res
