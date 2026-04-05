#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K


def lenet5(X):
    '''
    X is a K.Input of shape (m, 28, 28, 1)
    containing the input images for the network
    '''
    K.initializers.HeNormal(seed=0)
    model = K.Sequential()
    model.add(
        K.layers.Conv2D(6, kernel_size=(5, 5), activation="relu",
                        padding="same", kernel_initializer="he_normal",
                        input_shape=(28, 28, 1)))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Conv2D(16, kernel_size=(5, 5), activation="relu",
                       padding="valid", kernel_initializer="he_normal"))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(120, activation="relu"))
    model.add(K.layers.Dense(84, activation="relu"))
    model.add(K.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam",
                  metrics=["accuracy"])
    return model(X)
