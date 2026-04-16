#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K


def inception_block(A_prev, filters):
    '''filters is a tuple or list containing
        F1, F3R, F3,F5R, F5, FPP, respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1
            convolution before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1
            convolution before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1
            convolution after the max pooling'''
    F1, F3R, F3 = filters[0], filters[1], filters[2]
    F5R, F5, FPP = filters[3], filters[4], filters[5]
    layer1 = K.layers.Conv2D(filters=F1,
                             kernel_size=(1, 1),
                             padding='same',
                             activation="relu")(A_prev)

    layer2 = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation="relu")(A_prev)
    layer2 = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             padding='same',
                             activation="relu")(layer2)

    layer3 = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation="relu")(A_prev)
    layer3 = K.layers.Conv2D(filters=F5,
                             kernel_size=(5, 5),
                             padding='same',
                             activation="relu")(layer3)

    layer4 = K.layers.MaxPooling2D(pool_size=(3, 3))(A_prev)
    layer4 = K.layers.Conv2D(filters=FPP,
                             kernel_size=(1, 1),
                             padding='same',
                             activation="relu")(layer4)
    return K.layers.Concatenate(axis=-1)([layer1,
                                          layer2,
                                          layer3,
                                          layer4])
