#!/usr/bin/env python3
'''
This module takes a dataframe/
and add some changes
'''
import pandas as pd


def array(df):
    '''
    This method shows columns in ndarray/
    format
    '''
    df.tail(10).to_numpy()
