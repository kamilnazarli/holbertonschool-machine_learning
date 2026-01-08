#!/usr/bin/env python3
'''
Plotting a line graph
'''
import numpy as np
import matplotlib.pyplot as plt


def line():
    '''
    to plot a line graph
    '''
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, c='r')
