#!/usr/bin/env python3
'''Variational Autoencoder factory function'''
import tensorflow as tf
import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the input."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''
    This function creates a Variational Autoencoder.
    '''
    # --- Encoder ---
    global_input = keras.layers.Input(shape=(input_dims,))
    x = global_input
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
        
    z_mean = keras.layers.Dense(latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, name="z_log_var")(x)
    
    z = Sampling(name="z")([z_mean, z_log_var])
    
    # Must output: latent representation, mean, log variance
    encoder = keras.models.Model(inputs=global_input, 
                                 outputs=[z, z_mean, z_log_var], 
                                 name="encoder")

    # --- Decoder ---
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    y = decoder_input
    for layer in hidden_layers[::-1]:
        y = keras.layers.Dense(layer, activation="relu")(y)
    
    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.models.Model(inputs=decoder_input, 
                                 outputs=decoder_output, 
                                 name="decoder")

    # --- Full VAE Model ---
    sampled_z, _, _ = encoder(global_input)
    auto_output = decoder(sampled_z)
    auto = keras.models.Model(inputs=global_input, 
                             outputs=auto_output, 
                             name="autoencoder")

    # Calculate KL Divergence using standard tensorflow backend functions
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)
    auto.add_loss(kl_loss)

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto