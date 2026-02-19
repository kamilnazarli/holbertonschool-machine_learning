#!/usr/bin/env python3
'''module documented'''
import numpy as np


def specificity(confusion):
    '''function documented'''
    res = []
    for i in range(len(confusion)):
        TN, FP, spec = 0, 0, 0
        id = 0
        for j in range(len(confusion[i])):
            if i != id and j != id:
                TN = confusion[j][i]
            elif i != j:
                FP += confusion[j][i]
        spec = TN / (TN + FP)
        res.append(spec)
    return np.asarray(res)
