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
import matplotlib.gridspec as gridspec


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


def rotate(array_in,x_shift,y_shift):
   array_out = np.concatenate((array_in[x_shift:,:,:],array_in[:x_shift,:,:]),axis=0)
   return np.concatenate((array_out[:,x_shift:,:],array_out[:,:x_shift,:]),axis=1)


def apply_perturbation(y_data):
   x_data = apply_watermark(y_data)
   x_data = apply_noise(x_data)
   x_data = apply_compression(x_data)

   return x_data


def apply_watermark(x_data):
   mask_path = "C:/Coding/Watermark-2.0/Watermark_data/orig_mask.png"
   num_lines  = 768
   num_columns= 512
   if os.path.isfile(mask_path):
      im = Image.open(mask_path)
      im.thumbnail((num_lines,num_columns))
      general_mask = np.array(im, dtype = np.uint8)/255
      mask_positiv = general_mask > 0.5
      mask_negativ = np.invert(mask_positiv)

      x_data = x_data.numpy()
      for idx_pic in range(x_data.shape[0]):
         x_shift = np.random.randint(-20,20)
         y_shift = np.random.randint(-20,20)
         alpha = np.random.uniform(0.2,0.8)
         current_mask = rotate(general_mask,x_shift,y_shift)
         x_data_single_pic = x_data[idx_pic,:,:,:]
         x_data_single_pic = \
            mask_positiv*x_data_single_pic*(1-alpha) + mask_positiv*alpha + \
            mask_negativ*x_data_single_pic
         x_data_single_pic = np.clip(x_data_single_pic,0,1)
         x_data[idx_pic,:,:,:] = x_data_single_pic
      x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
   else:
      print('Error: Mask not found')

   return x_data


def apply_noise(x_data):
   noise_factor = np.random.uniform(0.05,0.2)
   x_data = x_data + noise_factor * tf.random.normal(shape=x_data.shape) 
   x_data = tf.clip_by_value(x_data, clip_value_min=0., clip_value_max=1.)
   return x_data


def apply_compression(x_data):
   x_data = x_data.numpy()
   for idx_pic in range(x_data.shape[0]):
      compression_level = np.random.randint(7,17)
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


def load_files(path_to_pickle,path,num_lines,num_columns):
   pickle_exists = os.path.isfile(path_to_pickle)
   if pickle_exists:
      print('pickle file exists')
      dataset = np.load(path_to_pickle)
      correct_shape = (dataset.shape[1]==num_lines and dataset.shape[2]==num_columns)
      if correct_shape:
         print('Loading from pickle-file')
         num_pic = dataset.shape[0]
      else:
         num_pic,dataset = load_pics_from_folder(path,num_lines_orig,num_columns_orig)
         np.save(path_to_pickle,dataset)
   else:
      num_pic,dataset = load_pics_from_folder(path,num_lines_orig,num_columns_orig)
      np.save(path_to_pickle,dataset)

   return num_pic,dataset


def show_examples(y_test_data,x_test_data,decoded_imgs):
   n = 6
   plt.figure(figsize=(20, 9))
   gs1 = gridspec.GridSpec(3,n)
   gs1.update(wspace=0.0,hspace=0.0)
   for i in range(n):
      # display original
      ax = plt.subplot(gs1[i])
      # ax = plt.subplot(3, n, i + 1)
      plt.title("original")
      plt.imshow(tf.squeeze(y_test_data[i]))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # display original + noise
      bx = plt.subplot(gs1[i+n])
      # bx = plt.subplot(3, n, i + n + 1)
      plt.title("pertubed")
      plt.imshow(tf.squeeze(x_test_data[i]))
      plt.gray()
      bx.get_xaxis().set_visible(False)
      bx.get_yaxis().set_visible(False)

      # display reconstruction
      cx = plt.subplot(gs1[i+2*n])
      # cx = plt.subplot(3, n, i + 2*n + 1)
      plt.title("reconstructed")
      plt.imshow(tf.squeeze(decoded_imgs[i]))
      plt.gray()
      cx.get_xaxis().set_visible(False)
      cx.get_yaxis().set_visible(False)
   plt.autoscale(tight=True)
   plt.show()
   print('figure showen')


def main():
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
   # directory1 = "C:/Coding/Watermark-2.0/datasets/PIRM_Self-Val_set/LR"
   # directory2 = "C:/Coding/Watermark-2.0/datasets/PIRM_Self-Val_set/MR"
   directory3 = "C:/Coding/Watermark-2.0/datasets/google"
   num_lines_orig = 512
   num_columns_orig = 768
   # path_to_pickle = 'datasets/PIRM_dataset.npy'
   path_to_pickle = 'datasets/google_512_768.npy'

   num_pic,full_data_set = load_files(path_to_pickle,directory3,num_lines_orig,num_columns_orig)
   input_shape = (num_lines_orig,num_columns_orig,3)

   print(num_pic)
   full_data_set = full_data_set/255
   y_train_data, y_test_data = np.split(full_data_set,[int(0.8*num_pic)],0)
   y_train_data = tf.convert_to_tensor(y_train_data, dtype=tf.float32)
   y_test_data = tf.convert_to_tensor(y_test_data, dtype=tf.float32)

   # Parameter for testing
   num_filter1 = 8
   num_filter2 = 16
   num_filter3 = 32
   # Parameter for final training. More research needed. Try dashboard!
   # num_filter1 = 64
   # num_filter2 = 128
   # num_filter3 = 256
   class Denoise(Model):
      def __init__(self):
         super(Denoise, self).__init__()
         self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape), 
            layers.Conv2D(num_filter1, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(num_filter2, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(num_filter3, (3,3), activation='relu', padding='same', strides=2)])

         self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(num_filter3, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(num_filter2, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(num_filter1, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

      def call(self, x):
         encoded = self.encoder(x)
         decoded = self.decoder(encoded)
         return decoded

   autoencoder = Denoise()

   autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
   x_test_data = apply_perturbation(y_test_data)

   batch_size = 1
   epochs = 5
   sub_epochs = 5
   loss = np.zeros((0))
   val_loss = np.zeros((0))
   for epoch in range(epochs): 
      x_train_data = apply_perturbation(y_train_data)
      history = autoencoder.fit(x_train_data, y_train_data,
                      epochs=sub_epochs,
                      shuffle=True,
                      batch_size=batch_size,
                      validation_data=(x_test_data, y_test_data))
      loss = np.concatenate((loss,np.asarray(history.history['loss'])),axis=0)
      val_loss = np.concatenate((val_loss,np.asarray(history.history['val_loss'])),axis=0) 

   print(loss)
   print(val_loss)

   epochs_array = range(epochs*sub_epochs)

   plt.figure()
   plt.plot(epochs_array, loss, 'r', label='Final training loss: ' \
            + str("{:.2e}".format(loss[-1]) ))
   plt.plot(epochs_array, val_loss, 'bo', label='Final validation loss: ' \
            + str("{:.2e}".format(val_loss[-1])  ))
   plt.title('Training and Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss Value')
   plt.xscale('log')
   plt.yscale('log')
   plt.ylim([0.001, 1])
   plt.legend()
   plt.show()

   autoencoder.encoder.summary()
   autoencoder.decoder.summary()

   print('before applaying encoder')
   encoded_imgs = autoencoder.encoder(x_test_data).numpy()
   decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

   show_examples(y_test_data,x_test_data,decoded_imgs)

if __name__ == '__main__':
    main()
