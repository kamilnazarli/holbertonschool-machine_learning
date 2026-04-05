#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K


def lenet5(X):
    '''
    X is a K.Input of shape (m, 28, 28, 1)
    containing the input images for the network
    '''
    initializer = K.initializers.HeNormal(seed=0)

    Z1 = K.layers.Conv2D(6, kernel_size=(5, 5), activation="relu",
                         padding="same", kernel_initializer=initializer,
                         input_shape=(28, 28, 1))(X)
    P1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Z1)
    Z2 = K.layers.Conv2D(16, kernel_size=(5, 5), activation="relu",
                         padding="valid",
                         kernel_initializer=initializer)(P1)
    P2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Z2)

    F = K.layers.Flatten()(P2)
    D1 = K.layers.Dense(120, activation="relu")(F)
    D2 = K.layers.Dense(84, activation="relu")(D1)
    output = K.layers.Dense(10, activation="softmax")(D2)
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer="adam",
                  metrics=["accuracy"])
    return model
