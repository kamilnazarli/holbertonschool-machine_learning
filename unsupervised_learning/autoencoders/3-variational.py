#!/usr/bin/env python3
'''Variational Autoencoder factory function'''
import keras
from keras import ops


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the input."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''
    This function creates a Variational Autoencoder.

    Args:
        input_dims(int): containing the dimensions input
        hidden_layers(list): containing the number of nodes
            for each hidden layer in the encoder, respectively
        latent_dims(int): containing the dimensions of the
            latent space representation

    Returns:
        encoder: the encoder model (outputs: z, z_mean, z_log_var)
        decoder: the decoder model
        auto: the full autoencoder model compiled with adam and BCE
    '''
    # --- 1. Encoder ---
    global_input = keras.layers.Input(shape=(input_dims,))
    x = global_input
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
        
    # mean and log variance use activation=None (default)
    z_mean = keras.layers.Dense(latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, name="z_log_var")(x)
    
    # Custom sampling layer
    z = Sampling(name="z")([z_mean, z_log_var])
    
    # Encoder outputs must be ordered: latent representation, mean, log variance
    encoder = keras.models.Model(inputs=global_input, 
                                 outputs=[z, z_mean, z_log_var], 
                                 name="encoder")

    # --- 2. Decoder ---
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    y = decoder_input
    for layer in hidden_layers[::-1]:
        y = keras.layers.Dense(layer, activation="relu")(y)
    
    # Last layer in the decoder uses sigmoid activation
    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    
    decoder = keras.models.Model(inputs=decoder_input, 
                                 outputs=decoder_output, 
                                 name="decoder")

    # --- 3. Full Autoencoder ---
    sampled_z, _, _ = encoder(global_input)
    auto_output = decoder(sampled_z)
    auto = keras.models.Model(inputs=global_input, 
                             outputs=auto_output, 
                             name="autoencoder")

    # --- 4. Custom KL Divergence Component Integration ---
    # Since standard 'binary_crossentropy' is explicitly requested during compile,
    # we inject the crucial KL Divergence penalty separately to retain the VAE structure.
    kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
    kl_loss = ops.mean(kl_loss)
    auto.add_loss(kl_loss)

    # Compile requirements: adam optimization and binary cross-entropy loss
    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto