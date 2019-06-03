import tensorflow as tf 
from keras.utils import plot_model

mnist =tf.keras.datasets.mnist #28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train= tf.keras.utils.normalize(x_train, axis=1)
x_test= tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 128 neurons in the layer with default activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #probability distribution

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy', #loss is degree of error
            metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.33, epochs=5, batch_size=5, verbose=0) #train the model

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc) #prints value loss and value accuracy

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict([x_test])

import numpy as np 
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
