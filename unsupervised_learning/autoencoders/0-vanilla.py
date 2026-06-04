#!/usr/bin/env python3
'''autoencoders'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''
    This function creates an autoencoder.

    Args:
        input_dims(int): containing the dimensions input
        hidden_layers(list): containing the number of nodes
            for each hidden layer in the encoder, respectively
        latent_dims(int): containing the dimensions of the
            latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model

    '''
    inputs = keras.Input(shape=(input_dims,))
    encoder = keras.models.Sequential()
    # [
    #     keras.layers.Input(shape=(input_dims,)),
    #     keras.layers.Dense()
    # ])
    global_input = keras.layers.Input(shape=(input_dims,))
    encoder.add(global_input)
    for l in hidden_layers:
        encoder.add(keras.layers.Dense(l, activation="relu"))
    decoder = keras.models.Sequential()
    decoder.add(keras.layers.Input(shape=(latent_dims,)))
    for i, l in enumerate(hidden_layers[::-1]):
        if(i != len(hidden_layers) - 1):
            decoder.add(keras.layers.Dense(l, activation="relu"))
        else:
            decoder.add(keras.layers.Dense(l, activation="sigmoid"))
    compressed = encoder(global_input)
    final = decoder(compressed)
    auto = keras.models.Model(inputs=global_input, outputs=final)
    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )
    return encoder, decoder, auto
