import tensorflow as tf
import datetime
import numpy as np

from tensorflow.keras import layers

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = tf.summary.create_file_writer(log_dir)


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))
x_train, x_test = x_train / 255.0, x_test / 255.0
# tf.summary.image('preprocess', x_train, 10)
# writer.flush()

# def create_model():
#   return tf.keras.models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
#   ])

# model = create_model()

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(28,28,1)))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.summary()
model.add(layers.MaxPooling2D(2,2))
model.summary()
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.summary()
model.add(layers.MaxPooling2D(2,2))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1), tf.keras.callbacks.EarlyStopping(patience=2,   restore_best_weights=True)]


model.fit(x=x_train, 
          y=y_train, 
          epochs=10, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
  images = np.reshape(x_train[0:10], (-1, 28, 28, 1))
  tf.summary.image("25 training data examples", images, max_outputs=10, step=5)