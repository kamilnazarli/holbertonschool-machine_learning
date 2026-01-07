#!/usr/bin/env python3
'''
This module creates a
slice function
'''


def slice(df):
    '''
    this method accepts df as a dataframe
    and returns every 60th rows
    '''
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
