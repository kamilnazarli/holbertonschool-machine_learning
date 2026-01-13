#!/usr/bin/env python3
'''Integration'''


def poly_integral(poly, C=0):
    ''' integration '''
    if type(poly) is not list or len(poly) == 0 or C is None:
        return None
    ls = []
    if len(poly) == 1 and poly[0] == 0:
        if C == 0:
            return [0]
        else:
            return [C]
    if len(poly) == 1:
        ls.append(poly[0])
        ls.insert(0, C)
        return ls
    for i in range(len(poly)):
        if poly[i]/(i+1) == poly[i]//(i+1):
            ls.append(poly[i]//(i+1))
        else:
            ls.append(poly[i]/(i+1))
    ls.insert(0, C)
    return ls
