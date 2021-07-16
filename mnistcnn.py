import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

x_train , x_test = x_train/255.0, x_test/255.0

print(y_train)

tf.random.set_seed(1)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2,   restore_best_weights=True)]

training_history = model.fit(x_train,y_train, epochs=20, batch_size=128, validation_split = 0.2, callbacks = callbacks, verbose = 1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("test data loss :",test_loss)
print("test data acc :",test_acc)


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