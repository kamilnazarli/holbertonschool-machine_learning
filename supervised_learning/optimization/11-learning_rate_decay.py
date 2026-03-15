#!/usr/bin/env python3
'''module documented'''
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''learning rate decay'''
    alpha = alpha / (1 + decay_rate * )
