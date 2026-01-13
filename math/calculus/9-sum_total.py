#!/usr/bin/env python3
'''Total sum'''


def summation_i_squared(n):
    ''' summation '''
    if n < 1:
        return None
    s = sum(map(lambda i: i**2, range(1, n+1)))
    return s
