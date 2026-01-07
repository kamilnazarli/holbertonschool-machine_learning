#!/usr/bin/env python3
'''
This module creates a
slice function
'''
import pandas as pd


def slice(df):
    '''
    this method accepts df as a dataframe
    and returns every 60th rows
    '''
    return df[['High', 'Low', 'Close', 'Volume_BTC']].iloc[::60]
