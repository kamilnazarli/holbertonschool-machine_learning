#!/usr/bin/env python3
'''Derivation list'''


def poly_derivative(poly):
    ''' derivative '''
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    ls = []
    for i in range(len(poly)):
        if i-1 >= 0:
            ls.append(poly[i]*i)
    return ls
