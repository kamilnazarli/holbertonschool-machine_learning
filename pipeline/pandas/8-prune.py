#!/usr/bin/env python3
'''
This module creates a function
that removes entries with NaN values
'''


def prune(df):
    '''
    drop entries with
    NaN values in close
    '''
    df.dropna(subset=['Close'], inplace=True)
    return df

