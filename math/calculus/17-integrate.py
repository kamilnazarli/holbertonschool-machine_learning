#!/usr/bin/env python3
'''Integration'''


def poly_integral(poly, C=0):
    ''' integration '''
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    ls = []
    for i in range(len(poly)):
        if poly[i]/(i+1) == poly[i]//(i+1):
            ls.append(poly[i]//(i+1))
        else:
            ls.append(poly[i]/(i+1))
    ls.insert(0, C)
    return ls
