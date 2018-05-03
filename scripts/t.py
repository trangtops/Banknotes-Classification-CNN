import time

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from matplotlib import pyplot as plt
import itertools

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from tensorflow.python.platform import gfile

import Data

def build_model(nfeatures, num_channels, num_classes, learning_rate):
    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=7, strides=2, padding='same', activation='relu',
                     input_shape=(nfeatures, num_channels)))
    model.add(BatchNormalization(axis=2))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization(axis=2))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))

    model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer
    # opt = keras.optimizers.Adadelta(lr=learning_rate)
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=False)
    # opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt,
                  metrics=['accuracy'])

    return model


model = build_model(500,32,5,1.)
print(model.summary())