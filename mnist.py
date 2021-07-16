import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

t1 = time.time()


tf.random.set_seed(1)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
], name = 'mnist_model')

print(model.summary())

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2,   restore_best_weights=True)]

training_history = model.fit(x_train,y_train, batch_size=128,   epochs=20, validation_split = 0.2, callbacks = callbacks, verbose = 1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("test data loss :",test_loss)
print("test data acc :",test_acc)

t2 = time.time()
print(f"process time : {t2-t1}")

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

