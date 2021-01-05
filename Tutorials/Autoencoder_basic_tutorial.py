import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

latent_dim = 64

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

num_plots = 7
for train_iter in range(num_plots):
    autoencoder.fit(x_train, x_train,
                    epochs=train_iter+1,
                    shuffle=True,
                    batch_size=600,
                    validation_data=(x_test, x_test))

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    num_subplots = 10
    plt.figure(figsize=(20, 4))
    for subplot_idx in range(num_subplots):
        # subplot_idx = i
        # display original
        ax = plt.subplot(2, num_subplots, subplot_idx + 1)
        plt.imshow(x_test[subplot_idx])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, num_subplots, subplot_idx + 1 + num_subplots)
        plt.imshow(decoded_imgs[subplot_idx])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(fname='Result after ' + str(train_iter) + ' iterations')
    # plt.show()

print('End of Basic Tutorial')