# Downloading Data
!git clone https://github.com/nirajsoft01/Track1.git
!pip3 install imgaug

# Importing Required Libreries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras as keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from google.colab import files
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

# Pandas Data CSV Insertion
datadir = 'Track1'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
data.head()
pd.set_option('display.max_colwidth', -1)
data.head()

# Remove unnecessory Path location

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

# Grouping or Binning the Dataset
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)

# Set a boundary limit for balanced -> not biased dataset
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

print('total data:', len(data))
remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)

data.drop(data.index[remove_list], inplace=True)

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))



# Train-Validation Split
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    # center image append
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_path.append(os.path.join(datadir,left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_path.append(os.path.join(datadir,right.strip()))
    steering.append(float(indexed_data[3])-0.15)

  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

# Image Augmentation Techniques - Generating new dataset from existing training dataset
# Zoom training images to create some new datasets
def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))         #scale upto 30% more
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)

# Horizontal or Vertical crop
def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)

# add brightness to image for genrate new training image
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))       # low brightness perform better in training process
    image = brightness.augment_image(image)
    return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)

# flip the image to genrate more balanced dataset, because our model is skwed to left rather than right
def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle          # also flip steering angles
    return image, steering_angle
random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)


def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle

# Image Preprocessing
def img_preprocess(img):
    # img = mpimg.imread(image)
    img = img[60:135,:,:]                      # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # NVIDIA R Color
    img = cv2.GaussianBlur(img,  (3, 3), 0)    # Smoothing Image
    img = cv2.resize(img, (200, 66))           # resize
    img = img/255                              # convert b/w 0--1
    return img

# Batch Genrator - memory efficient
def batch_generator(image_paths, steering_ang, batch_size, istraining):

  while True:
    batch_img = []
    batch_steering = []

    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)

      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]

      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

# NVIDIA MODEL
def nvidia_model():
      model = Sequential()
      model.add(Convolution2D(24, kernel_size=(5,5),  strides=(2,2),                                                             input_shape=(66,200,3),activation='elu'))
      model.add(Convolution2D(36, kernel_size=(5,5),   strides=(2,2), activation='elu'))
      model.add(Convolution2D(48, kernel_size=(5,5), strides=(2,2), activation='elu'))
      model.add(Convolution2D(64, kernel_size=(3,3), activation='elu'))
      model.add(Convolution2D(64, kernel_size=(3,3), activation='elu'))
      #model.add(Dropout(0.5))


      model.add(Flatten())

      model.add(Dense(100, activation='elu'))
      #model.add(Dropout(0.5))

      model.add(Dense(50, activation='elu'))
      #model.add(Dropout(0.5))

      model.add(Dense(10, activation ='elu'))
      #model.add(Dropout(0.5))

      model.add(Dense(1))

      optimizer= Adam(learning_rate=1e-4)   #1e-3 = 0.001   1e-4 = 0.0004
      model.compile(loss='mse', optimizer=optimizer)

      return model

model = nvidia_model()

#Training The Model
#history = model.fit(X_train, y_train, epochs= 30, validation_data=(X_valid, y_valid), batch_size=100, verbose=1, shuffle=1)

history = model.fit(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=15,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)

# Analysing the Performance of model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

# Saving the Model into PC
model.save('model.h5')
files.download('model.h5')
