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

HEIGHT = 128
WIDTH = 128

num = int(args[1])
epoch_num = (num-1)*10 + 20
print(f"num : {num}")
SAVE_PATH = f"./savedmodels/n_{num}/"

close_train_path = glob.glob(f"../datasets/holograms/train/fft_close_holo/num_{(num-1):05}/*.png")
far_train_path = glob.glob(f"../datasets/holograms/train/fft_far_holo/num_{num:05}/*.png")
close_test_path = glob.glob(f"../datasets/holograms/test/fft_close_holo/num_{(num-1):05}/*.png")
far_test_path = glob.glob(f"../datasets/holograms/test/fft_far_holo/num_{num:05}/*.png")

close_train_path = close_train_path[0:3000]
far_train_path = far_train_path[0:3000]

print(len(close_train_path))
print(len(far_train_path))
print(len(close_test_path))
print(len(far_test_path))

#IMAGE ARRAY INITIALIZING
train_image = []
train_label = []
test_image = []
test_label = []

#INPUTTING IMG TO ARRAY
for ii in close_train_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    train_image.append(img)
    train_label.append(1)

for ii in far_train_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    train_image.append(img)
    train_label.append(0)

for ii in close_test_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    test_image.append(img)
    test_label.append(1)

for ii in far_test_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    test_image.append(img)
    test_label.append(0)

train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
test_image = np.asarray(test_image)
test_label = np.asarray(test_label)

#NORMALIZATION, ARRAY CASTING FOR GPU CALCULATION
train_image = train_image.astype('float32')
train_image = train_image / 255.0
test_image = test_image.astype('float32')
test_image = test_image / 255.0

#DATASETS SHUFFLING
for ii in [train_image, train_label, test_image, test_label]:
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
callbacks = callbacks.EarlyStopping(patience=epoch_num//3, restore_best_weights=True)

#TRAINING
training_history = model.fit(train_image, train_label, epochs=epoch_num, batch_size=64, validation_split = 0.2, callbacks = callbacks, verbose = 1)

os.makedirs(SAVE_PATH, exist_ok=True)
model.save(SAVE_PATH)
#EVALUATION WITH TEST DATA
test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)

print("test data loss :",test_loss)
print("test data acc :",test_acc)