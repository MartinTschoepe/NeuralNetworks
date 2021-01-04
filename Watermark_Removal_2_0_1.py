import numpy as np
import os
import io
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_pics_from_folder(directory,num_lines,num_columns):
    print('Loading from folder')
    num_pic = 0
    pic_list = []
    for filename in os.listdir(directory):
        filename = directory + "/" + filename
        if num_pic%100 == 0:
            print(num_pic)
        if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
            im = Image.open(filename)
            num_pic+=1
            new_pic = np.array(im, dtype = np.int32)
            new_pic_cropped =  new_pic[:num_lines,:num_columns,:].reshape((1,num_lines,num_columns,3))
            pic_list.append(new_pic_cropped)
        if num_pic >= 200: break
    full_data_set = np.vstack(pic_list)
    return num_pic,full_data_set

def apply_perturbation(y_data,noise_factor=0.2,compression_level=10):
    use_noise       = False
    use_compression = True
    use_watermark   = False
    if use_noise:
        x_data = apply_noise(y_data,noise_factor)
    else:
        x_data = y_data
    if use_compression:
        x_data = apply_compression(x_data,compression_level)
    # if use_watermark:
    #     x_data,y_data = apply_watermark(x_data,y_data)

    return x_data

def apply_noise(y_data,noise_factor):
    x_data = y_data + noise_factor * tf.random.normal(shape=y_data.shape) 
    x_data = tf.clip_by_value(x_data, clip_value_min=0., clip_value_max=1.)
    return x_data

def apply_compression(x_data,compression_level):
    x_data = x_data.numpy()
    for idx_pic in range(x_data.shape[0]):
        output = io.BytesIO()
        x_data_single_pic = x_data[idx_pic,:,:,:]
        x_data_single_pic = (x_data_single_pic*255).astype(np.uint8)
        x_data_single_pic = Image.fromarray(x_data_single_pic)
        x_data_single_pic.save(output, format='jpeg', quality=compression_level)
        x_data_single_pic = Image.open(output)
        x_data_single_pic = np.array(x_data_single_pic, dtype=np.float32)/255
        x_data[idx_pic,:,:,:] = x_data_single_pic
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    return x_data


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    # directory1 = "C:/Coding/Watermark-2.0/datasets/PIRM_Self-Val_set/LR"
    # directory2 = "C:/Coding/Watermark-2.0/datasets/PIRM_Self-Val_set/MR"
    directory3 = "C:/Coding/Watermark-2.0/datasets/google"
    num_lines_orig = 512
    num_columns_orig = 768
    # path_to_pickle = 'datasets/PIRM_dataset.npy'
    path_to_pickle = 'datasets/google_512_768.npy'
    pickle_exists = os.path.isfile(path_to_pickle)
    if pickle_exists:
        print('pickle file exists')
        full_data_set = np.load(path_to_pickle)
        correct_shape = (full_data_set.shape[1]==num_lines_orig and full_data_set.shape[2]==num_columns_orig)
        if correct_shape:
            print('Loading from pickle-file')
            num_pic = full_data_set.shape[0]
        else:
            num_pic,full_data_set = load_pics_from_folder(directory3,num_lines_orig,num_columns_orig)
            np.save(path_to_pickle,full_data_set)
    else:
        num_pic,full_data_set = load_pics_from_folder(directory3,num_lines_orig,num_columns_orig)
        np.save(path_to_pickle,full_data_set)

    input_shape = (num_lines_orig,num_columns_orig,3)
    print(num_pic)
    full_data_set = full_data_set/255
    y_train_data, y_test_data = np.split(full_data_set,[int(0.8*num_pic)],0)
    y_train_data = tf.convert_to_tensor(y_train_data, dtype=tf.float32)
    y_test_data = tf.convert_to_tensor(y_test_data, dtype=tf.float32)

    scale_factor1 = 1
    scale_factor2 = 2
    batch_size = 1
    class Denoise(Model):
      def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=input_shape), 
          layers.Conv2D(16*scale_factor1, (3,3), activation='relu', padding='same', strides=2),
          layers.Conv2D(8*scale_factor2, (3,3), activation='relu', padding='same', strides=2)
          ])

        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(8*scale_factor2, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(16*scale_factor1, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    autoencoder = Denoise()

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    noise_factor = 0.01
    compression_level = 5 # Higher is better quality.
    x_test_data = apply_perturbation(y_test_data,noise_factor,compression_level)

    num_epochs = 1
    for epoch in range(num_epochs): 
        x_train_data = apply_perturbation(y_train_data,noise_factor,compression_level)
        # noise_factor = 0.2
        # x_train_data = y_train_data + noise_factor * tf.random.normal(shape=y_train_data.shape) 
        # x_train_data = tf.clip_by_value(x_train_data, clip_value_min=0., clip_value_max=1.)
        # print((255*y_train_data[1]).numpy().astype("uint8"))
        autoencoder.fit(x_train_data, y_train_data,
                        epochs=1,
                        shuffle=True,
                        batch_size=batch_size,
                        validation_data=(x_test_data, y_test_data))

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    print('before applaying encoder')
    encoded_imgs = autoencoder.encoder(x_test_data).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    print('after applaying encoder')
    n = 4
    plt.figure(figsize=(15, 6))
    for i in range(n):

        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(x_test_data[i]))
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
    print('figure showen')

if __name__ == '__main__':
    main()
