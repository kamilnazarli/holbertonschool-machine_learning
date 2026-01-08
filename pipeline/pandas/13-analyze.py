#!/usr/bin/python3
'''
This module analyze dataframe
'''


def analyze(df):
    '''
    This function accepts
    dataframe and show
    descriptive statistics
    '''
    return df[[col for col in df.columns if col != 'Timestamp']].describe()
