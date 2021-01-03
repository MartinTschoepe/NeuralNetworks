import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
x_train = list_ds.skip(val_size)
x_test = list_ds.take(val_size)

# batch_size   = 32
batch_size = 8
# batch_size = 1
# img_height   = 180
# img_width    = 180
num_channels = 3
# img_height = 511
# img_width = 768
img_height   = 100
img_width    = 100
input_shape = (img_width, img_height, num_channels)

x_train_numpy = np.zeros((1,1))
x_test_numpy = np.zeros((1,1))

AUTOTUNE = tf.data.experimental.AUTOTUNE

x_train = x_train.cache().prefetch(buffer_size=AUTOTUNE)
x_test = x_test.cache().prefetch(buffer_size=AUTOTUNE)

num_fig_train = tf.data.experimental.cardinality(x_train).numpy()
num_batches_train = int(num_fig_train/batch_size)
print("num_batches_train= " + str(num_batches_train))
num_fig_test = tf.data.experimental.cardinality(x_test).numpy()
num_batches_test = int(num_fig_test/batch_size)
print("num_batches_test= " + str(num_batches_test))

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img

x_train = x_train.map(process_path, num_parallel_calls=AUTOTUNE)
x_test = x_test.map(process_path, num_parallel_calls=AUTOTUNE)


for image in x_train.take(1):
  print("Image shape: ", image.numpy().shape)


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

x_train= configure_for_performance(x_train)
x_test = configure_for_performance(x_test)

# for counter in range(num_batches_train):
image_batch_x_train = next(iter(x_train))
# for counter in range(num_batches_test):
image_batch_x_test = next(iter(x_test))

# plt.figure(figsize=(10, 10))
# for i in range(9):
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(image_batch_x_train[i].numpy().astype("uint8"))
#   plt.axis("off")
# plt.show()

print(image_batch_x_train.numpy().shape)
print(image_batch_x_test.numpy().shape)
image_batch_x_train =image_batch_x_train/255
image_batch_x_test =image_batch_x_test/255

noise_factor = 0.2
x_train
x_train_noisy = image_batch_x_train + noise_factor * tf.random.normal(shape=image_batch_x_train.numpy().shape)
x_test_noisy = image_batch_x_test + noise_factor * tf.random.normal(shape=image_batch_x_test.numpy().shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

#TODO: Dont forget to normalize.
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

scale_factor = 1
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=input_shape), 
      layers.Conv2D(16*scale_factor, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8*scale_factor, (3,3), activation='relu', padding='same', strides=2)
      ])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8*scale_factor, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16*scale_factor, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# autoencoder.fit(image_batch_x_train, image_batch_x_train,
autoencoder.fit(x_train_noisy, image_batch_x_train,
# autoencoder.fit(x_train, x_train,
                epochs=300,
                shuffle=True,
                batch_size=batch_size,
                # validation_data=(x_test, x_test))
                # validation_data=(image_batch_x_test, image_batch_x_test))
                validation_data=(x_test_noisy, image_batch_x_test))

autoencoder.encoder.summary()

autoencoder.decoder.summary()

print(image_batch_x_train.shape)
encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
print(decoded_imgs.shape)

n = 5
plt.figure(figsize=(5, 2))
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


