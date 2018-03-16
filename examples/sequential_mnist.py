from __future__ import print_function
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from ind_rnn import IndRNNCell, RNN

batch_size = 100
num_classes = 10
epochs = 200
hidden_units = 128

learning_rate = 1e-3

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

cells = [IndRNNCell(hidden_units),
         IndRNNCell(hidden_units)]

print('Evaluate IRNN...')
model = Sequential()
model.add(RNN(cells,  input_shape=x_train.shape[1:]))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

rmsprop = Adam(lr=learning_rate, amsgrad=True)

model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint('weights/imdb_indrnn_mnist.h5', monitor='val_acc',
                                     save_best_only=True, save_weights_only=True, mode='max')])

model.load_weights('weights/imdb_indrnn_mnist.h5')

scores = model.evaluate(x_test, y_test, verbose=0)
print('IndRNN test score:', scores[0])
print('IndRNN test accuracy:', scores[1])
