#!/usr/bin/env python3
'''
This module creates a function
to set a column as index
'''


def index(df):
    '''
    to set 'Timestamp' column
    as index column
    '''
    df.set_index(keys=['Timestamp'], inplace=True)
    return df
