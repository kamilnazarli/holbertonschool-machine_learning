#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def residual_block(A_prev, f, k, s=1):
    '''A_prev - the output of previous layer
       f - number of filters
       k - kernel size
       s - stride
    '''
    initializer = K.initializers.HeNormal(seed=0)
    X = K.layers.Conv2D(filters=f,
                        kernel_size=k,
                        strides=(s, s),
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    X = K.layers.Conv2D(filters=f,
                        kernel_size=k,
                        strides=(1, 1),
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    return X


def resnet50():
    '''func builds the ResNet-50 architecture'''
    initializer = K.initializers.HeNormal(seed=0)
    input = K.Input(shape=(224, 224, 3))
    X = K.layers.Conv2D(
                        filters=64,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        kernel_initializer=initializer)(input),
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2))(X)
    res = residual_block(X, 64, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)    

    res = residual_block(X, 64, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 64, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 128, 3, 2)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 128, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 128, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 128, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 256, 3, 2)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 256, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 256, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 256, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 256, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 256, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 512, 3, 2)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 512, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    res = residual_block(X, 512, 3)
    X = K.layers.Add()([res, X])
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation("relu")(X)

    X = K.layers.GlobalAveragePooling()(X)
    output = K.layers.Dense(1000, activation="softmax")(X)
    model = K.Model(input, output)
    return model
