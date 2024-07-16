#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Тренировка keras модели

Created on Tue Jul 16 13:51 2024

@author: gnv
"""
import os
import gc
import time

import tensorflow as tf
from tensorflow.keras.models import load_model

import logging

from helpers import setup_logger

from nds import construct_training_data_set
from nds_model_k import model

if __name__ == "__main__":
    log = setup_logger('', '_nds_train.out', console_out=True)
else:
    log = logging.getLogger(__name__)

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
MODEL_FILE = 'nds_model.keras'
MODEL_PATH = os.path.join(DATA_PATH, MODEL_FILE)


def train(_nn=1000, _i_start=0, _n_epochs=100, _batch_size=10):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        _dev_name = 'GPU'
        log.info("We got a GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        _dev_name = 'CPU'
        log.info("Sorry, no GPU for you...")

    log.info(f"******** NDS keras model training on {_dev_name} ********")
    _start = time.time()

    _xt, _yt = construct_training_data_set(_nn, _i_start)

    if os.path.isfile(MODEL_PATH):
        w_model = load_model(MODEL_PATH)
        log.info(f"Model was loaded from the {MODEL_PATH}")
    else:
        w_model = model

    # Fit the model
    w_model.fit(_xt, _yt, epochs=_n_epochs, batch_size=_batch_size)
    w_model.save(MODEL_PATH, overwrite=True)
    log.info(f"Model was saved to the {MODEL_PATH}")

    # Evaluate the model
    scores = w_model.evaluate(_xt, _yt)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    _dur = (time.time() - _start)  # sec
    log.info(f"Duration: {_dur:.2f} sec")


if __name__ == "__main__":
    import sys

    nn_to_train = 1000
    i_start = 0
    n_epochs = 3
    n_args = len(sys.argv)
    if n_args > 1:
        nn_to_train = int(sys.argv[1])
        if n_args > 2:
            i_start = int(sys.argv[2])
            if n_args > 3:
                n_epochs = int(sys.argv[3])

    train(nn_to_train, i_start, n_epochs)

    sys.exit(0)

