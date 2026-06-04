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
    global_input = keras.layers.Input(shape=(input_dims,))
    x = global_input
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
    latent_space = keras.layers.Dense(latent_dims, activation="relu")(x)
    encoder = keras.models.Model(inputs=global_input, outputs=latent_space)

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    y = decoder_input
    for layer in hidden_layers[::-1]:
        y = keras.layers.Dense(layer, activation="relu")(y)
    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)

    auto_output = decoder(encoder(global_input))
    auto = keras.models.Model(inputs=global_input, outputs=auto_output)
    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )
    return encoder, decoder, auto
