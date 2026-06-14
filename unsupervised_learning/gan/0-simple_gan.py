import tensorflow as tf
from tensorflow import keras
import numpy as np

class Simple_GAN(keras.Model):

    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        super().__init__()                                
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter

        self.learning_rate    = learning_rate
        self.beta_1           = .5                                 
        self.beta_2           = .9                                 

        # Using standard BinaryCrossEntropy for both
        # Set from_logits=True if your discriminator's last layer DOES NOT have a sigmoid activation
        self.loss_fn = keras.losses.BinaryCrossEntropy(from_logits=False)
        
        self.d_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)

    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, data):
        # Keras .fit() passes data here, but since we use self.get_real_sample(), 
        # we can ignore the incoming 'data' argument.

        # ----------------------------------------
        # 1. Train the Discriminator (disc_iter times)
        # ----------------------------------------
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            
            with tf.GradientTape() as d_tape:
                # We need the generator inside the tape if we want gradients to flow, 
                # but for the discriminator alone, we can just pass the generated output.
                fake_sample = self.get_fake_sample(training=True)
                
                # Get discriminator predictions
                real_predictions = self.discriminator(real_sample, training=True)
                fake_predictions = self.discriminator(fake_sample, training=True)
                
                # Calculate loss: real should be close to 1, fake close to 0
                real_loss = self.loss_fn(tf.ones_like(real_predictions), real_predictions)
                fake_loss = self.loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
                discr_loss = real_loss + fake_loss

            # Calculate and apply gradients for Discriminator
            d_gradients = d_tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # ----------------------------------------
        # 2. Train the Generator (1 time)
        # ----------------------------------------
        with tf.GradientTape() as g_tape:
            # Generate fake samples and pass them through the updated discriminator
            fake_sample = self.get_fake_sample(training=True)
            fake_predictions = self.discriminator(fake_sample, training=True)
            
            # The generator wants the discriminator to think these are REAL (target = 1)
            gen_loss = self.loss_fn(tf.ones_like(fake_predictions), fake_predictions)

        # Calculate and apply gradients for Generator
        g_gradients = g_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        # Return metrics to be displayed by Keras progress bar
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}