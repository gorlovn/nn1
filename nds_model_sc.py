#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sckit-Learn Keras модель
"""
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

DS_NN = int(os.environ.get('DS_NN', '14646'))


def create_model():
    # Don't use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Initialize the model
    _model = Sequential()

    # Add the input layer
    _model.add(Dense(units=DS_NN, activation='relu', input_shape=(DS_NN, )))

    # Add the output layer
    _model.add(Dense(units=DS_NN, activation='sigmoid'))

    # Compile the model
    _model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return _model


if __name__ == "__main__":

    model = create_model()
    # Summary of the model
    model.summary()
