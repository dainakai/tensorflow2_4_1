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

#RANDOM SEED CONSTRAINT
# def set_seed(seed=200):
#     tf.random.set_seed(seed)

#     # optional
#     # for numpy.random
#     np.random.seed(seed)
#     # for built-in random
#     random.seed(seed)
#     # for hash seed
#     os.environ["PYTHONHASHSEED"] = str(seed)
# set_seed(SEED)

print("seed unset")

num = int(args[1])
# atp = int(args[2])
epoch_num = num*30
print(f"num : {num}")
# print(f"attempt : {atp}")
# SAVE_PATH = f"./savedmodels/light/n_{num}/{atp:02}/"
# TXT_PATH = f"./saveddata//light/n_{num}/{atp:02}.txt"

#ALL IMAGES PATH LISTS FOR TRAINING AND TEST
# close_train_path = glob.glob(f"./datasets/holograms/train/fft_close_holo/num_{num:05}/*.png")
# far_train_path = glob.glob(f"./datasets/holograms/train/fft_far_holo/num_{num:05}/*.png")
# close_test_path = glob.glob(f"./datasets/holograms/test/fft_close_holo/num_{num:05}/*.png")
# far_test_path = glob.glob(f"./datasets/holograms/test/fft_far_holo/num_{num:05}/*.png")

close_train_path = glob.glob(f"./datasets/holograms/train/close_holo/num_{(num-1):05}/*.bmp")
far_train_path = glob.glob(f"./datasets/holograms/train/far_holo/num_{num:05}/*.bmp")
close_test_path = glob.glob(f"./datasets/holograms/test/close_holo/num_{(num-1):05}/*.bmp")
far_test_path = glob.glob(f"./datasets/holograms/test/far_holo/num_{num:05}/*.bmp")

print(len(close_train_path))
print(len(far_train_path))
print(len(close_test_path))
print(len(far_test_path))

close_train_path = close_train_path[0:3000]
far_train_path = far_train_path[0:3000]
# close_test_path = close_train_path[0:100]
# far_test_path = far_train_path[0:100]

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

#MODEL
# model = models.Sequential([
#     layers.Conv2D(32, (5,5), activation='relu', input_shape=(HEIGHT,WIDTH,1)),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (5,5), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='relu'),
#     layers.Dense(2, activation='softmax')
# ])

# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape=(HEIGHT,WIDTH,1), kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3,3), kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(32, (3,3), kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3,3), kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3,3), kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(1000, kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(100, kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10, kernel_regularizer='l2'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('softmax'))

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

#VGG16
# model = Sequential()
# model.add(Conv2D(64, (3,3), padding='same', input_shape=(HEIGHT,WIDTH,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(128, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(256, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(256, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(256, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(512, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(512, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(512, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(512, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(512, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(512, (3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(4096))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(1000))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('softmax'))

# model = Sequential()
# model.add(Conv2D(32, (5,5), input_shape=(HEIGHT,WIDTH,1), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(32, (3,3), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(64), kernel_regularizer=regularizers.l2(0.01))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10), kernel_regularizer=regularizers.l2(0.01))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(BatchNormalization())
# model.add(Activation('softmax'))

# model = Sequential()
# model.add(Flatten(input_shape=(HEIGHT,WIDTH,1)))
# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(BatchNormalization())
# model.add(Activation('softmax'))



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

# model.save(SAVE_PATH)
#EVALUATION WITH TEST DATA
test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)

print("test data loss :",test_loss)
print("test data acc :",test_acc)

#PLOT
# y = training_history.history['loss']
# x = range(1,len(y)+1)
# plt.semilogy(x,y,label="loss for training")

# y = training_history.history['val_loss']
# plt.semilogy(x,y,linestyle="dashed" ,label="loss for validation")

# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Loss value [-]")
# plt.xticks(np.arange(1,len(y)+1,1))
# plt.xlim(0,len(y)+1)
# plt.savefig("lossDuringTraining.png")

# plt.clf()

# y = training_history.history['accuracy']
# plt.plot(x,y,label="accuracy for training")

# y = training_history.history['val_accuracy']
# plt.plot(x,y,linestyle="dashed", label = "accuracy for validation")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy value")
# plt.xticks(np.arange(1,len(y)+1,1))
# plt.xlim(0,len(y)+1)
# plt.savefig("accuracyDuringTraining.png")

# file = open(TXT_PATH,"w")
# file.write(f"test data loss : {test_loss}\n")
# file.write(f"test data accyracy : {test_acc}")
# file.write("\n\n")
# file.write("training history accuracy\n")
# file.write('\n'.join([str(n) for n in training_history.history['accuracy']]))
# file.write("\n\n")
# file.write("training history val accuracy\n")
# file.write('\n'.join([str(n) for n in training_history.history['val_accuracy']]))
# file.write("\n\n")
# file.write("training history loss\n")
# file.write('\n'.join([str(n) for n in training_history.history['loss']]))
# file.write("\n\n")
# file.write("training history val loss\n")
# file.write('\n'.join([str(n) for n in training_history.history['val_loss']]))
# file.close()