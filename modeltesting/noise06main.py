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
from tensorflow.keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten, Dropout, MaxPooling2D

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

#ALL IMAGES PATH LISTS FOR TRAINING AND TEST
close_train_path = glob.glob("../datasets/holograms/train/fft_close_holo/num_00006/*")
far_train_path = glob.glob("../datasets/holograms/train/fft_far_holo/num_00006/*")
close_test_path = glob.glob("../datasets/holograms/test/fft_close_holo/num_00006/*")
far_test_path = glob.glob("../datasets/holograms/test/fft_far_holo/num_00006/*")

close_train_path = close_train_path[0:3000]
far_train_path = far_train_path[0:3000]
# close_test_path = close_train_path[0:100]
# far_test_path = far_train_path[0:100]

print(f"Close datasets:{len(close_train_path)}")
print(f"Far datasets:{len(far_train_path)}")
print(f"Close test datasets:{len(close_test_path)}")
print(f"Far test datasets:{len(far_test_path)}")

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


model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(HEIGHT,WIDTH,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# model = Sequential()
# model.add(Conv2D(32, (5,5), input_shape=(HEIGHT,WIDTH,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(BatchNormalization())
# model.add(Activation('softmax'))

# model = Sequential()
# model.add(Conv2D(32, (5,5), input_shape=(HEIGHT,WIDTH,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10))
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
callbacks = callbacks.EarlyStopping(patience=100, restore_best_weights=True)

#TRAINING
training_history = model.fit(train_image, train_label, epochs=30, batch_size=64, validation_split = 0.2, callbacks = callbacks, verbose = 1)

#EVALUATION WITH TEST DATA
test_loss, test_acc = model.evaluate(test_image, test_label, verbose=0)

print("test data loss :",test_loss)
print("test data acc :",test_acc)

#PLOT
y = training_history.history['loss']
x = range(1,len(y)+1)
plt.semilogy(x,y,label="loss for training")

y = training_history.history['val_loss']
plt.semilogy(x,y,linestyle="dashed" ,label="loss for validation")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss value [-]")
plt.xticks(np.arange(1,len(y)+1,1))
plt.xlim(0,len(y)+1)
plt.savefig("lossDuringTraining.png")

plt.clf()

y = training_history.history['accuracy']
plt.plot(x,y,label="accuracy for training")

y = training_history.history['val_accuracy']
plt.plot(x,y,linestyle="dashed", label = "accuracy for validation")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy value")
plt.xticks(np.arange(1,len(y)+1,1))
plt.xlim(0,len(y)+1)
plt.savefig("accuracyDuringTraining.png")

file = open("testResult.txt","w")
file.write(f"test data loss : {test_loss}\n")
file.write(f"test data accyracy : {test_acc}")
file.close()

