#!/usr/bin/env python3
'''
This module creates function
which sorts and transposes the dataframe
'''


def flip_switch(df):
    '''
    This function accepts
    dataframe and sort it
    prints as transposed
    '''
    return df.sort_values(by='Timestamp', ascending=False).T
