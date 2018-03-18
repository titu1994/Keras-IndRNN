'''
Addition problem. Code reused from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py

'''
from __future__ import print_function
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import initializers
from keras.callbacks import ModelCheckpoint

from ind_rnn import IndRNNCell, RNN

if not os.path.exists('weights'):
    os.makedirs('weights/')

# Parameters taken from https://arxiv.org/abs/1803.04831 ; (above)
TIME_STEPS = 100
NUM_UNITS = 128
LEARNING_RATE = 0.0002
STEPS_PER_EPOCH = 100
NUM_EPOCHS = 600

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50


# Code reused from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py
def batch_generator():
    while True:
        """Generate the adding problem dataset"""
        # Build the first sequence
        add_values = np.random.rand(BATCH_SIZE, TIME_STEPS)

        # Build the second sequence with one 1 in each half and 0s otherwise
        add_indices = np.zeros_like(add_values)
        half = int(TIME_STEPS / 2)
        for i in range(BATCH_SIZE):
            first_half = np.random.randint(half)
            second_half = np.random.randint(half, TIME_STEPS)
            add_indices[i, [first_half, second_half]] = 1

        # Zip the values and indices in a third dimension:
        # inputs has the shape (batch_size, time_steps, 2)
        inputs = np.dstack((add_values, add_indices))
        targets = np.sum(np.multiply(add_values, add_indices), axis=1)
        yield inputs, targets

cells = [IndRNNCell(NUM_UNITS),
         IndRNNCell(NUM_UNITS)]

print('Build model...')
model = Sequential()
model.add(RNN(cells, input_shape=(TIME_STEPS, 2)))
model.add(Dense(1, activation='linear', bias_initializer=initializers.constant(0.1)))

# try using different optimizers and different optimizer configs
optimizer = Adam(LEARNING_RATE, amsgrad=True)
model.compile(loss='mse', optimizer='adam')

model.fit_generator(batch_generator(), steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=NUM_EPOCHS, verbose=1,
                    callbacks=[ModelCheckpoint('weights/imdb_indrnn_addition.h5', monitor='loss',
                                               save_best_only=True, save_weights_only=True, mode='min')])
