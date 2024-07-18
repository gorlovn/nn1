#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
MODEL_FILE = 'nds_model.keras'
MODEL_PATH = os.path.join(DATA_PATH, MODEL_FILE)
MODEL_JSON_FILE = 'nds_model.json'
MODEL_JSON_PATH = os.path.join(DATA_PATH, MODEL_JSON_FILE)
MODEL_WEIGHTS_FILE = 'nds_model.weights.h5'
MODEL_WEIGHTS_PATH = os.path.join(DATA_PATH, MODEL_WEIGHTS_FILE)
MODEL_ARCHITECTURE_KEY = os.environ.get('KERAS_MODEL_ARCHITECTURE_KEY', 'keras_model_architecture')
MODEL_WEIGHTS_KEY = os.environ.get('KERAS_MODEL_WEIGHTS_KEY', 'keras_model_weights')
REDIS_EX = int(os.environ.get('KERAS_MODEL_REDIS_EX', '1800'))
RAM_DATA_PATH = '/tmp/ramdisk'
RAM_MODEL_PATH = os.path.join(RAM_DATA_PATH, MODEL_FILE)
