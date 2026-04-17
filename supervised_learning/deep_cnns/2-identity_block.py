#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K


def inception_block(A_prev, filters):
    '''filters is a tuple or list 
       containing F11, F3, F12, respectively:
       F11 is the number of filters in
       the first 1x1 convolution
       F3 is the number of filters in
       the 3x3 convolution
       F12 is the number of filters in
       the second 1x1 convolution'''
    F11, F3, F12 = filters[0], filters[1], filters[2]
    layer1 = K.layers.Conv2D(filters=F11,
                             kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=K.initializers.HeNormal(seed=0))(A_prev)
    layer1 = K.layers.BatchNormalization()(layer1)
    layer1 = K.layers.ReLU()(layer1)

    layer2 = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=K.initializers.HeNormal(seed=0))(layer1)
    layer2 = K.layers.BatchNormalization()(layer2)
    layer2 = K.layers.ReLU()(layer2)

    layer3 = K.layers.Conv2D(filters=F12,
                             kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=K.initializers.HeNormal(seed=0))(layer2)
    layer3 = K.layers.BatchNormalization()(layer3)

    res = K.layers.Concatenate(axis=-1)([layer3, A_prev])
    res_A = K.layers.ReLU()(res)
    return res_A
