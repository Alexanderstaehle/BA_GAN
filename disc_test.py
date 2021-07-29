import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from lstmwgan import WGANGP

latent_dim = 100
n_samples = 30
n_epochs = 40

if __name__ == '__main__':
    real_trajectories = np.load('data/preprocessed/validation_gl.npy', allow_pickle=True)
    real_trajectories = real_trajectories[:n_samples]
    gen_inputs = []
    random_latent_vectors = noise = tf.random.normal((tf.shape(real_trajectories)[0], latent_dim), 0, 1)
    gen_inputs.append(real_trajectories)
    gen_inputs.append(random_latent_vectors)
    gan = WGANGP()
    gan.generator.load_weights('parameters/G_model_gl_' + str(n_epochs) + '.h5')  # params/G_model_100.h5
    generated_trajectories = gan.generator(gen_inputs).numpy()
    gan.discriminator.load_weights('parameters/D_model_gl_' + str(n_epochs) + '.h5')  # params/G_model_100.h5
    predict_real = gan.discriminator(real_trajectories)
    predict_fake = gan.discriminator(generated_trajectories)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(range(0, n_samples), predict_real, color='r')
    ax.scatter(range(0, n_samples), predict_fake, color='b')
    plt.show()
