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

test_num = int(args[1])
itr = 1
SAVE_PATH = f"../full_model/"

close_test_path = glob.glob(f"../datasets/holograms/test/fft_close_holo/num_{(test_num-1):05}/*.png")
far_test_path = glob.glob(f"../datasets/holograms/test/fft_far_holo/num_{test_num:05}/*.png")

print(len(close_test_path))
print(len(far_test_path))

#IMAGE ARRAY INITIALIZING

test_image = []
test_label = []

#INPUTTING IMG TO ARRAY
for ii in close_test_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    test_image.append(img)
    test_label.append(1)

for ii in far_test_path:
    img = img_to_array(load_img(ii, target_size=(HEIGHT,WIDTH), grayscale=True))
    test_image.append(img)
    test_label.append(0)

test_image = np.asarray(test_image)
test_label = np.asarray(test_label)

#NORMALIZATION, ARRAY CASTING FOR GPU CALCULATION
test_image = test_image.astype('float32')
test_image = test_image / 255.0


#DATASETS SHUFFLING
# for ii in [test_image, test_label]:
#     np.random.seed(1)
#     np.random.shuffle(ii)

model = models.load_model(SAVE_PATH)
#EVALUATION WITH TEST DATA
# test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)

# print("test data loss :",test_loss)
# print("test data acc :",test_acc)

# file = open("accsave.txt","a")
# file.write(f"{num}\t{test_acc}\n")
# file.close()

for idx in range(itr):
    for ii in [test_image, test_label]:
        np.random.seed(idx)
        np.random.shuffle(ii)
    test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)
    print("test data loss :",test_loss)
    print("test data acc :",test_acc)
    file = open("full_accsave.txt","a")
    file.write(f"{test_num}\t{test_acc}\n")
    file.close()
