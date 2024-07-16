#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Lesson 04
First Neural Net in Keras

Created on Tue Jul 16 13:27 2024

@author: gnv
"""
import os
import gc
import time

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import logging

from helpers import setup_logger

if __name__ == "__main__":
    log = setup_logger('', '_k4.out', console_out=True)
else:
    log = logging.getLogger(__name__)

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
DATA_FILE = 'pima-indians-diabetes.data.csv'
DATA_PATH = os.path.join(DATA_PATH, DATA_FILE)

# Load the dataset
dataset = np.loadtxt(DATA_PATH, delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]
# Define and Compile
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
