#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_prediction(_input_vector, _weights=weights_1, _bias=bias):
    layer_1 = np.dot(_input_vector, _weights) + _bias
    layer_2 = sigmoid(layer_1)
    return layer_2


def step(_input_vector, _weights, _bias, _target):
    _prediction = make_prediction(_input_vector, _weights, _bias)
    _mse = np.square(_prediction - _target)
    _derivative = 2 * (_prediction - _target)
    print(f"Prediction: {_prediction}; Error: {_mse}; Derivative: {_derivative}")

    return _derivative

target = 0
input_vector = np.array([2, 1.5])
derivative = step(input_vector, weights_1, bias, target)

weights_1 = weights_1 - derivative
print(weights_1)
derivative = step(input_vector, weights_1, bias, target)
