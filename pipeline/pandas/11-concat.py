#!/usr/bin/env python3
'''
This module creates a function
to concatenate 2 dataframes
'''
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    '''
    this function concatenates
    two dataframes
    '''
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index<=1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
