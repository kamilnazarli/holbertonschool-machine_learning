#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def l2_reg_cost(cost, model):
    '''cost calculation'''
    return cost + tf.add_n(model.losses)
