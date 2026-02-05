#!/usr/bin/env python3
'''module documented'''


class Binomial:
    '''class documented'''
    def __init__(self, data=None, n=1, p=0.5):
        '''constructor documented'''
        self.data = data
        self.n = n
        self.p = p
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if not(p >= 0 and p <= 1):
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data) # mean = n * p
            self.var = self.variance() # variance = n * p * (1 - p)
            self.p = 1 - (self.var / self.mean)
            self.n = round(self.mean / self.p)

    def variance(self):
        '''variance documented'''
        return (sum((x - self.mean) ** 2 for x in self.data) /
                (len(self.data) - 1))
