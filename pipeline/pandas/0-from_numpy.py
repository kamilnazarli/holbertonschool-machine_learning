#This module creates a method that converts array into dataframe
#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    '''This method is used to change /
    numpy array into pandas dataframe
    '''
    start = 65
    cols = []
    for i in range(26):
        cols.append(chr(start+i))
    new_df = pd.DataFrame(array, columns=cols[:array.shape[1]])
    return new_df
