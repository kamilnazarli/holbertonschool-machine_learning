#!/usr/bin/env python3

def from_numpy(array):
    '''This method is used to change /
    numpy array into pandas dataframe
    '''
    start = 65
    cols = []
    for i in range(26):
    cols.append(chr(start+i))
    return pd.DataFrame(array, columns = cols[:array.shape[1]])
