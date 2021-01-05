import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

def plot_in_window(fig):
    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(fig[i]))
        plt.gray()
    plt.show()
    return

def apply_padding(pad_size,x_train,x_test):
    padding = tf.constant([[0, 0], [ pad_size,  pad_size], [ pad_size,  pad_size], [0, 0]])
    x_train   = tf.pad(x_train, padding, "CONSTANT")
    x_test    = tf.pad(x_test , padding, "CONSTANT")
    return pad_size,x_train,x_test

def split_and_reshape_tf_tensor(tensor,grid_size_x,grid_size_y):
    vertial_grid = tf.split(tensor,grid_size_x,axis=0)
    tensor = layers.Concatenate(axis=1)(vertial_grid)
    horizontal_grid = tf.split(tensor,grid_size_y,axis=0)
    tensor = layers.Concatenate(axis=2)(horizontal_grid)
    print(tensor.shape)

    return tensor

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print(x_train.shape)

    grid_size_x = 25
    grid_size_y = 20
    x_train = split_and_reshape_tf_tensor(x_train,grid_size_x,grid_size_y)
    x_train = layers.Concatenate(axis=3)(tf.split(x_train,3,axis=0)) #3 Channels
    x_test = split_and_reshape_tf_tensor(x_test,grid_size_x,grid_size_y)
    x_test = layers.Concatenate(axis=3)( (x_test,x_test,x_test) ) #3 Channels

    print('x_train: ') 
    print(x_train.shape)
    print('x_test: ')  
    print(x_test.shape)

    pad_size,x_train,x_test = apply_padding(0,x_train,x_test)

    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

    # plot_in_window(x_test_noisy)
    pic_size_x = 28*grid_size_x
    input_size_x = pic_size_x + pad_size*2
    pic_size_y = 28*grid_size_y
    input_size_y = pic_size_y + pad_size*2
    num_channels = 3
    input_shape = (input_size_x, input_size_y, num_channels)

    class Denoise(Model):
      def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=input_shape), 
          layers.Conv2D(16*2, (3,3), activation='relu', padding='same', strides=2),
          layers.Conv2D(8*2, (3,3), activation='relu', padding='same', strides=2)
          ])

        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(8*2, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(16*2, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    autoencoder = Denoise()

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    print(x_train.shape)
    print(input_shape)
    batch_size = 1
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=10,
                    shuffle=True,
                    batch_size=batch_size,
                    validation_data=(x_test_noisy, x_test))

    autoencoder.encoder.summary()

    autoencoder.decoder.summary()

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 1
    plt.figure(figsize=(grid_size_x/5, grid_size_y/2))
    for i in range(n):

        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(x_test_noisy[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_imgs[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    main()
