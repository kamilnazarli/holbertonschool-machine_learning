#!/usr/bin/env python3
'''
This module renames a/
specific column in dataframe
'''
import pandas as pd


def rename(df):
    '''This function accepts a dataframe/
     renames the column and returns it
    '''

    df = df.rename(columns={'Timestamp':'Datetime'})
    df = pd.to_datetime(df)
    return df[['Datetime', 'Close']]
