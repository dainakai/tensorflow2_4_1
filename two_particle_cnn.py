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

#TENSORBOARD OUTPUTTING DIRECTORY
log_dir = "logs/two_particle_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#CONSTANTS
HEIGHT = 128 #MODEL INPUT IMAGE HEIGHT
WIDTH = 128 #MODEL INPUT IMAGE WIDTH
C_TRAIN_COUNT = 3000 #CLOSE DROPLETS IMAGES FOR TRAIN
F_TRAIN_COUNT = 3000 #FAR DROPLETS IMAGES FOR TRAIN
C_TEST_COUNT = 500 #CLOSE DROPLETS IMAGES FOR TEST
F_TEST_COUNT = 500 #FAR DROPLETS IMAGES FOR TEST
SEED = 1

#RANDOM SEED CONSTRAINT
def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
set_seed(SEED)

#ALL IMAGES PATH LISTS FOR TRAINING AND TEST
close_train_path = glob.glob("./datasets/fft_two_particle_datasets_close/*")
far_train_path = glob.glob("./datasets/fft_two_particle_datasets_far/*")
close_test_path = glob.glob("./datasets/fft_two_particle_datasets_close_test/*")
far_test_path = glob.glob("./datasets/fft_two_particle_datasets_far_test/*")

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
    np.random.seed(SEED)
    np.random.shuffle(ii)

#MODEL
model = models.Sequential([
    layers.Conv2D(32, (5,5), activation='relu', input_shape=(HEIGHT,WIDTH,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(2, activation='softmax')
])

#MODEL SUMMARY OUTPUT
print(model.summary())

#OPTIMIZATION OPTIONS
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#EARLY-STOPPING SETTING AND PREPARE FOR VISUALIZATION WITH TENSORBOARD
callbacks = [callbacks.EarlyStopping(patience=2, restore_best_weights=True),
             callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

#TRAINING
training_history = model.fit(train_image, train_label, epochs=20, batch_size=64, validation_split = 0.2, callbacks = callbacks, verbose = 1)

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

#INTERMEDIATE DATA PREPARERING FOR TENSORBOARD
def scaler(array):
    m = array.min()
    M = array.max()
    array = (array-m)/(M-m)
    return array

intermediate_data_1 = img_to_array(load_img("./datasets/fft_two_particle_datasets_close_test/00128.png", target_size=(HEIGHT,WIDTH), grayscale=True))
intermediate_data_1 = np.asarray(intermediate_data_1)
intermediate_data_1 = intermediate_data_1.astype('float32') / 255.0
intermediate_data_1 = intermediate_data_1.reshape([1,HEIGHT,WIDTH,1])

intermediate_data_0 = img_to_array(load_img("./datasets/fft_two_particle_datasets_far_test/00000.png", target_size=(HEIGHT,WIDTH), grayscale=True))
intermediate_data_0 = np.asarray(intermediate_data_0)
intermediate_data_0 = intermediate_data_0.astype('float32') / 255.0
intermediate_data_0 = intermediate_data_0.reshape([1,HEIGHT,WIDTH,1])

t_1_answer = model.predict(intermediate_data_1)
t_0_answer = model.predict(intermediate_data_0)
print("t=1 model answer :",t_1_answer)
print("t=0 model answer :",t_0_answer)

#INTERMEDIATE LAYERS SETTINGS
conv2d0 = "conv2d"
intermediate_conv2d0_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(conv2d0).output)
conv2d0_output_1 = intermediate_conv2d0_model.predict(intermediate_data_1)
conv2d0_output_0 = intermediate_conv2d0_model.predict(intermediate_data_0)
conv2d0_output_1 = scaler(conv2d0_output_1)
conv2d0_output_0 = scaler(conv2d0_output_0)

conv2d1 = "conv2d_1"
intermediate_conv2d1_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(conv2d1).output)
conv2d1_output_1 = intermediate_conv2d1_model.predict(intermediate_data_1)
conv2d1_output_0 = intermediate_conv2d1_model.predict(intermediate_data_0)
conv2d1_output_1 = scaler(conv2d1_output_1)
conv2d1_output_0 = scaler(conv2d1_output_0)

maxpooling = "max_pooling2d"
intermediate_maxpooling_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(maxpooling).output)
maxpooling_output_1 = intermediate_maxpooling_model.predict(intermediate_data_1)
maxpooling_output_0 = intermediate_maxpooling_model.predict(intermediate_data_0)
maxpooling_output_1 = scaler(maxpooling_output_1)
maxpooling_output_0 = scaler(maxpooling_output_0)

#TENSORBOARD SETTINGS
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    tf.summary.image("10 training data examples", train_image, max_outputs=10, step=1)

    conv2d_weights = model.get_layer("conv2d").get_weights()[0] # height? width? channels filters
    images_conv2d = conv2d_weights.transpose(3,0,1,2) # filters height width channels
    images_conv2d = scaler(images_conv2d)
    tf.summary.image("conv2d", images_conv2d, max_outputs=32, step = 0)

    conv2d_1_weights = model.get_layer("conv2d_1").get_weights()[0]
    images_conv2d_1 = conv2d_1_weights.transpose(3,0,1,2)
    images_conv2d_1 = scaler(images_conv2d_1)
    tf.summary.image("conv2d_1", images_conv2d_1[:,:,:,1].reshape([64,3,3,1]), max_outputs=64, step = 0)

    tf.summary.image("intermediate t=1 data examples", intermediate_data_1, max_outputs=1, step=0)
    tf.summary.image("intermediate t=0 data examples", intermediate_data_0, max_outputs=1, step=0)
    tf.summary.image("intermediate conv2d0 t=1 layer output", 255.0*conv2d0_output_1.transpose(3,1,2,0), max_outputs=100, step = 0)
    tf.summary.image("intermediate conv2d0 t=0 layer output", 255.0*conv2d0_output_0.transpose(3,1,2,0), max_outputs=100, step = 0)

    tf.summary.image("intermediate maxpooling t=1 layer output", 255.0*maxpooling_output_1.transpose(3,1,2,0), max_outputs=100, step = 0)
    tf.summary.image("intermediate maxpooling t=0 layer output", 255.0*maxpooling_output_0.transpose(3,1,2,0), max_outputs=100, step = 0)
    
    tf.summary.image("intermediate conv2d1 t=1 layer output", 255.0*conv2d1_output_1.transpose(3,1,2,0), max_outputs=100, step = 0)
    tf.summary.image("intermediate conv2d1 t=0 layer output", 255.0*conv2d1_output_0.transpose(3,1,2,0), max_outputs=100, step = 0)