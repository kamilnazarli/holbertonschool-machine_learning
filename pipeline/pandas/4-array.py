#!/usr/bin/env python3
'''
This module takes a dataframe/
and add some changes
'''


def array(df):
    '''
    This method shows columns in ndarray/
    format
    '''
    return df.tail(10).to_numpy()
