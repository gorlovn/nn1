#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras модель
"""
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

DS_NN = int(os.environ.get('DS_NN', '14646'))

# Don't use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize the model
model = Sequential()

# Add the input layer
model.add(Dense(units=DS_NN, activation='relu', input_shape=(DS_NN, )))

# Add the output layer
model.add(Dense(units=DS_NN, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


if __name__ == "__main__":
    # Summary of the model
    model.summary()
