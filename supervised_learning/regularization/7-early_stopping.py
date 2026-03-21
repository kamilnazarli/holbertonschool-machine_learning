#!/usr/bin/env python3
'''module documented'''
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """early stopping"""
    if cost + threshold > opt_cost:
        count += 1
    else:
        opt_cost = cost + threshold
        count = 0

    if count >= patience:
        return True, count
    else:
        return False, count
