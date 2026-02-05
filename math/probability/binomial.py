#!/usr/bin/env python3
'''module documented'''


class Binomial:
    '''class documented'''
    def __init__(self, data=None, n=1, p=0.5):
        '''constructor documented'''
        self.data = data
        if data is None:
            self.n = n
            self.p = p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (p > 0 and p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.var = (sum((x - self.mean) ** 2 for x in self.data) /
                        len(self.data))
            self.p = 1 - (self.var / self.mean)
            self.n = round(self.mean / self.p)
            self.p = self.mean / self.n

    def pmf(self, k):
        '''prob mass function for binomial'''
        if not (isinstance(k, int)):
            k = int(k)
        if k < 0:
            return 0
        return ((self.factorial(self.n) /
                 ((self.factorial(k) * self.factorial(self.n - k))) *
                 (self.p ** k) * (1 - self.p) ** (self.n - k)))

    @staticmethod
    def factorial(n):
        '''factorial documented'''
        fact = 1
        for i in range(1, n + 1):
            fact = fact * i
        return fact
