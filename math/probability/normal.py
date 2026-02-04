#!/usr/bin/env python3
'''module documented'''


class Normal:
    '''class documented'''
    def __init__(self, data=None, mean=0., stddev=1.):
        '''constructor documented'''
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = mean
            self.stddev = stddev
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = self.standard_dev()

    def z_score(self, x):
        '''method1 documented'''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''method2 documented'''
        return z * self.stddev + self.mean

    def pdf(self, x):
        '''method3 documented'''
        return ((1 / (2 * 3.1415926536 * self.stddev ** 2) ** (0.5)) 
                * 2.7182818285 ** (-(x-self.mean) ** 2 / 2 * self.stddev ** 2))
    
    def standard_dev(self):
        '''method documented'''
        stddev = 0
        for i in range(len(self.data)):
            stddev = stddev + (self.data[i] - self.mean) ** 2
        stddev = (stddev / len(self.data)) ** (0.5)
        return stddev
