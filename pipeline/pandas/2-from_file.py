#!/usr/bin/env python3
'''
This module loads data from a/
specific file
'''
import pandas as pd


def from_file(filename, delimiter):
    '''
    This function accepts/
    file and read it with/
    specific delimiter
    '''
    return pd.read_csv(filename, delimiter=delimiter)
