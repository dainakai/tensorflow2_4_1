'''
PROGRAM NAME: Droplets Proximity Detection on 2-D Spectrum Distribution of Hologram Image
                 with Convolutional Neural Network (CNN)
AUTHOR: Dai NAKAI
DATE: 2021/5/16

This program detects whether two droplets, which are on the hologram plane are close (approaching) or far through Machine Learning Model called Convolutional Neural Network.
Holograms will be processed into spectrum distribution images, and model obtains
predictions as probability to be close (corresponds to 1) and far (0).

'''
import tensorflow as tf 
from tensorflow.keras import models, layers , callbacks
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import glob
import numpy as np
import datetime
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten, Dropout, MaxPooling2D, ActivityRegularization
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Precision, Recall, Accuracy
import cv2
import sys
args = sys.argv

#TENSORBOARD OUTPUTTING DIRECTORY
log_dir = "logs/noise_first/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#CONSTANTS
HEIGHT = 128 #MODEL INPUT IMAGE HEIGHT
WIDTH = 128 #MODEL INPUT IMAGE WIDTH
C_TRAIN_COUNT = 1000 #CLOSE DROPLETS IMAGES FOR TRAIN
F_TRAIN_COUNT = 1000 #FAR DROPLETS IMAGES FOR TRAIN
C_TEST_COUNT = 500 #CLOSE DROPLETS IMAGES FOR TEST
F_TEST_COUNT = 500 #FAR DROPLETS IMAGES FOR TEST
SEED = 12345

mnum = 10

#IMAGE ARRAY INITIALIZING
train_image = []
train_label = []

for num in range(2,mnum):
    close_train_path = glob.glob(f"./datasets/holograms/train/close_holo/num_{(num-1):05}/*.bmp")
    far_train_path = glob.glob(f"./datasets/holograms/train/far_holo/num_{num:05}/*.bmp")
    close_train_path = close_train_path[0:3000]
    far_train_path = far_train_path[0:3000]

    for ii in close_train_path:
        img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
        train_image.append(img)
        train_label.append(1)

    for ii in far_train_path:
        img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
        train_image.append(img)
        train_label.append(0)

train_image = np.asarray(train_image)
train_label = np.asarray(train_label)

#NORMALIZATION, ARRAY CASTING FOR GPU CALCULATION
train_image = train_image.astype('float32')
train_image = train_image / 255.0

#DATASETS SHUFFLING
for ii in [train_image, train_label]:
    np.random.seed(1)
    np.random.shuffle(ii)

#VGG Custom #real
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', input_shape=(HEIGHT,WIDTH,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

#MODEL SUMMARY OUTPUT
print(model.summary())

#OPTIMIZATION OPTIONS
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#EARLY-STOPPING SETTING AND PREPARE FOR VISUALIZATION WITH TENSORBOARD
callbacks = callbacks.EarlyStopping(patience=30, restore_best_weights=True)

#TRAINING
training_history = model.fit(train_image, train_label, epochs=150, batch_size=64, validation_split = 0.2, callbacks = callbacks, verbose = 1)

model.save("./full_model/")