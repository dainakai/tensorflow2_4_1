import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt

iris = load_iris()
# tf.random.set_seed(1)

random.seed(12345)
Ndata = len(iris.data)
idxr = [k for k in range(Ndata)]
random.shuffle(idxr)
Ndata_train = int (Ndata*0.5)
train_data = iris.data[idxr[:Ndata_train]]
train_labels = iris.target[idxr[:Ndata_train]]

val_data = iris.data[idxr[Ndata_train:]]
val_labels = iris.target[idxr[Ndata_train:]]

model = keras.models.Sequential([
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='SGD',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

training_history = model.fit(train_data,train_labels,
                             validation_data=(val_data,val_labels),
                             epochs = 20,
                             batch_size = Ndata_train//10,
                             verbose = 1)

y = training_history.history['loss']
x = range(len(y))
plt.semilogy(x,y,label="loss for training")
#
y = training_history.history['val_loss']
x = range(len(y))
plt.semilogy(x,y,label="loss for validation", alpha = 0.5)
#
plt.legend()
plt.xlabel("Steps")
plt.show()
#---------
y = training_history.history['accuracy']
x = range(len(y))
plt.plot(x,y,label="accuracy for training")
y = training_history.history['val_accuracy']
x = range(len(y))
plt.plot(x,y,label = "accuracy for validation")
plt.legend()
plt.xlabel("Steps")
plt.ylim(0,1.1)
plt.show()

