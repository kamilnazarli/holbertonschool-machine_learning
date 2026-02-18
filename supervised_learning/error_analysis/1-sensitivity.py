#!/usr/bin/env python3
'''module documented'''


def sensitivity(confusion):
    '''function documented'''
    TP = sum([confusion[i][j] 
              for i in range(len(confusion)) 
              for j in range(len(confusion[i])) if i==j])
    FN = sum([confusion[i][j] 
              for i in range(len(confusion)) 
              for j in range(len(confusion[i])) if i!=j])
    return TP / (TP + FN)
