'''
PROGRAM NAME: Droplets Proximity Detection on 2-D Spectrum Distribution of Hologram Image
                 with Convolutional Neural Network (CNN) , SEGMENTATION
AUTHOR: Dai NAKAI
DATE: 2021/5/16

This program detects whether two droplets, which are on the hologram plane are close (approaching) or far through Machine Learning Model called Convolutional Neural Network.
Holograms will be processed into spectrum distribution images, and model obtains
predictions as probability to be close (corresponds to 1) and far (0).

Input images are splitted and processed fft on each segments.

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

#CONSTANTS
HEIGHT = 128 #MODEL INPUT IMAGE HEIGHT
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

args = sys.argv
opt = int(args[1])
segnum = int(args[2])
pnum = int(args[3])

epoch_num = 150
if opt == 0:
    count = segnum*segnum
    bs = 8
elif opt == 1:
    count = segnum*segnum + 2*segnum*(segnum-1)
    bs = 8
elif opt == 2:
    count = segnum*segnum + 2*segnum*(segnum-1) + (segnum-1)*(segnum-1)
    bs = 4

WIDTH = HEIGHT*count
SAVE_PATH = f"./segmodels/opt{opt}/segnum{segnum}/num_{pnum:05}/"
TXT_PATH = f"./segdata/opt{opt}/segnum{segnum}/num_{pnum:05}/"
os.makedirs(SAVE_PATH,exist_ok=True)
os.makedirs(TXT_PATH,exist_ok=True)

close_train_path = glob.glob(f"/media/dai/DATADISK/two_particle_datasets/segmentation/train/close_holo/opt{opt}/segnum{segnum}/num_{(pnum-1):05}/*")
far_train_path = glob.glob(f"/media/dai/DATADISK/two_particle_datasets/segmentation/train/far_holo/opt{opt}/segnum{segnum}/num_{pnum:05}/*")

print(len(close_train_path))
print(len(far_train_path))

close_train_path = close_train_path[0:3000]
far_train_path = far_train_path[0:3000]

#IMAGE ARRAY INITIALIZING
train_image = []
train_label = []

#INPUTTING IMG TO ARRAY
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

#MODEL

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

callbacks = [callbacks.EarlyStopping(patience=30, restore_best_weights=True), callbacks.ModelCheckpoint(filepath=SAVE_PATH, monitor='val_loss', verdose = 1, save_best_only=True)]

#TRAINING
training_history = model.fit(train_image, train_label, epochs=epoch_num, batch_size=bs, validation_split = 0.2, callbacks = callbacks, verbose = 1)

model.save(SAVE_PATH)

file = open(TXT_PATH+"lossacc.txt","w")
file.write("training history accuracy\n")
file.write('\n'.join([str(n) for n in training_history.history['accuracy']]))
file.write("\n\n")
file.write("training history val accuracy\n")
file.write('\n'.join([str(n) for n in training_history.history['val_accuracy']]))
file.write("\n\n")
file.write("training history loss\n")
file.write('\n'.join([str(n) for n in training_history.history['loss']]))
file.write("\n\n")
file.write("training history val loss\n")
file.write('\n'.join([str(n) for n in training_history.history['val_loss']]))
file.close()