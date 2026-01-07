#!/usr/bin/env python3
'''
This module creates a function
that sorts values in descending order
'''


def high(df):
    '''
    Accepts dataframe and sorts
    High column values
    '''
    return df.sort_values(by='High', ascending=False)
