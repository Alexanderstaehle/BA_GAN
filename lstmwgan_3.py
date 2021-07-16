from __future__ import print_function, division

from abc import ABC

import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.core.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.initializers.initializers_v2 import HeUniform
from tensorflow.python.keras.layers import LeakyReLU, LSTM, Dropout, Lambda
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l1

SAMPLE_INTERVAL = 1
BATCH_SIZE = 32
EPOCHS = 3

d_loss_values, g_loss_values = list(), list()


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated trajectory samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((1, 144, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP(keras.Model, ABC):
    def __init__(self):
        super(WGANGP, self).__init__()
        self.max_length = 144
        self.features = 1
        self.traj_shape = (self.max_length, self.features)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_discriminator = 1
        self.gp_weight = 10

        # Build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, real_trajectories, fake_trajectories):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated trajectory
        and added to the discriminator loss.
        """
        # Get the interpolated trajectory
        interpolated_traj = RandomWeightedAverage()([real_trajectories, fake_trajectories])

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_traj)
            # 1. Get the discriminator output for this interpolated trajectory.
            pred = self.discriminator(interpolated_traj, training=True)
        # 2. Calculate the gradients w.r.t to this interpolated trajectory.
        grads = gp_tape.gradient(pred, [interpolated_traj])[0]

        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def build_generator(self):
        model = Sequential()

        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.traj_shape), activation='tanh'))
        model.add(Reshape(self.traj_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        traj = model(noise)

        return Model(noise, traj)

    def build_discriminator(self):
        i = Input(shape=self.traj_shape,
                  name='input')
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
        d = Dense(units=64, use_bias=True, activation='relu',
                  kernel_initializer=HeUniform(seed=1),
                  name='embedding')
        dense_latlon = [d(x) for x in unstacked]
        e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(e)
        d = Dense(units=100, use_bias=True, activation='relu',
                  kernel_initializer=HeUniform(seed=1),
                  name='emb_trajpoint')
        dense_outputs = [d(x) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
        # LSTM Modeling Layer (many-to-one)
        lstm_cell = LSTM(units=100, recurrent_regularizer=l1(0.02))(emb_traj)
        dropout = Dropout(0.2)(lstm_cell)
        # Output
        tanh = Dense(1, activation='tanh')(dropout)
        model = Model(inputs=i, outputs=tanh)
        model.summary()
        return model

    def train_step(self, real_trajectories):
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        for _ in range(self.n_discriminator):
            # Get the latent vector
            noise = tf.random.normal((tf.shape(real_trajectories)[0], self.latent_dim), 0, 1)
            with tf.GradientTape() as tape:
                # Generate fake trajectories from the latent vector
                fake_trajectories = self.generator(noise, training=True)
                # Get the logits for the fake trajectories
                fake_logits = self.discriminator(fake_trajectories, training=True)
                # Get the logits for the real trajectories
                real_logits = self.discriminator(real_trajectories, training=True)

                # Calculate the discriminator loss using the fake and real trajectory logits
                d_cost = self.d_loss_fn(real_traj=real_logits, fake_traj=fake_logits)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(real_trajectories, fake_trajectories)

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight
                d_loss_values.append(d_loss)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        print("discriminator training done")
        # Train the generator
        # Get the latent vector
        noise = tf.random.normal((tf.shape(real_trajectories)[0], self.latent_dim), 0, 1)
        with tf.GradientTape() as tape:
            # Generate fake trajectories using the generator
            generated_trajectories = self.generator(noise, training=True)
            # Get the discriminator logits for fake trajectories
            gen_traj_logits = self.discriminator(generated_trajectories, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_traj_logits)
            g_loss_values.append(g_loss)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        print("gen done")
        return {"d_loss": d_loss, "g_loss": g_loss}


# Define the loss function for the discriminator.
def discriminator_loss(real_traj, fake_traj):
    real_loss = tf.reduce_mean(real_traj)
    fake_loss = tf.reduce_mean(fake_traj)
    return fake_loss - real_loss


# Define the loss function for the generator.
def generator_loss(fake_traj):
    return -tf.reduce_mean(fake_traj)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs={}):
        if epoch % SAMPLE_INTERVAL == 0:
            self.model.generator.save_weights("parameters/G_model_" + str(epoch) + ".h5")
            self.model.discriminator.save_weights("parameters/D_model_" + str(epoch) + ".h5")


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    callback = GANMonitor()
    plotCallback = PlotLearning()
    wgan = WGANGP()

    generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    # Training data
    X_train = np.load('data/preprocessed/train.npy', allow_pickle=True)
    wgan.fit(X_train, batch_size=32, epochs=1, callbacks=[plotCallback, callback])
