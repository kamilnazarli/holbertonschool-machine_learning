#!/usr/bin/env python3
'''module documented'''
import numpy as np


def specificity(confusion):
    '''function documented'''
    res = []
    for i in range(len(confusion)):
        TN, FP, spec = 0, 0, 0
        act_neg = 0
        for j in range(len(confusion[i])):
            if i != j:
                act_neg += confusion[i][j]
                FP += confusion[j][i]
        TN = act_neg - FP
        spec = TN / (TN + FP)
        res.append(spec)
    return np.asarray(res)
