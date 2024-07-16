#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Для заданного пациента
сформировать предполагаемые диагнозы ДН
используя сохраненную keras модель

Created on Tue Jul 16 14:54 2024

@author: gnv
"""
import os
import numpy as np
import redis
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

import logging

from helpers import setup_logger

from nds import get_diagnosis_dict
from nds_train_k import MODEL_PATH, MODEL_JSON_PATH, MODEL_WEIGHTS_PATH
from nds_train_k import init_device

from nds_predict import get_input_layer_dd, get_output_layer_ce

if __name__ == "__main__":
    log = setup_logger('', '_nds_predict_k.out', console_out=True)
else:
    log = logging.getLogger(__name__)

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

MODEL_ARCHITECTURE_KEY = os.environ.get('KERAS_MODEL_ARCHITECTURE_KEY', 'keras_model_architecture')
MODEL_WEIGHTS_KEY = os.environ.get('KERAS_MODEL_WEIGHTS_KEY', 'keras_model_weights')
REDIS_EX = int(os.environ.get('KERAS_MODEL_REDIS_EX', '1800'))


def load_k_model(_model_path, _save_to_redis=False):
    # Retrieve the model architecture and weights from Redis
    model_json_b = r.get(MODEL_ARCHITECTURE_KEY)
    model_json = None if model_json_b is None else model_json_b.decode('utf-8')
    model_h5 = r.get(MODEL_WEIGHTS_KEY)
    if model_json is not None and model_h5 is not None:
        # Create a new model from the architecture
        _model = model_from_json(model_json)
        # Load the weights into the new model
        _model.load_weights(model_h5)
        log.info("++++++++ We have got keras model from redis")
        return _model

    log.info(f"++++++++ Loading keras model from {_model_path}")
    _model = load_model(_model_path)
    _model.make_predict_function()
    _model.summary()  # Optional: Include this line to display the model structure when the model is reloaded

    if _save_to_redis:
        model_json = _model.to_json()  # Convert the model architecture to JSON
        log.info(f"Saving model architecture to {MODEL_JSON_PATH}")
        with open(MODEL_JSON_PATH, 'w') as f:
            f.write(model_json)

        log.info(f"Saving model weights to {MODEL_WEIGHTS_PATH}")
        # Save the weights of the model
        _model.save_weights(MODEL_WEIGHTS_PATH)

        # Load the model architecture and weights
        with open(MODEL_JSON_PATH, 'r') as f:
            model_json = f.read()

        with open(MODEL_WEIGHTS_PATH, 'rb') as f:
            model_h5 = f.read()

        log.info("Storing model to redis")
        # Store the model architecture in Redis
        r.set(MODEL_ARCHITECTURE_KEY, model_json, ex=REDIS_EX)

        # Store the model weights in Redis
        r.set(MODEL_WEIGHTS_KEY, model_h5, ex=REDIS_EX)

    return _model


def main(_p_id):

    log.info(f"******** Predict DN diagnosis list for the patient {_p_id} ********")

    _ds_dict = get_diagnosis_dict()
    if _ds_dict is None:
        return None
    _ds_keys = list(_ds_dict.keys())
    _ds_nn = len(_ds_keys)

    _il_dict = get_input_layer_dd(_p_id, _ds_dict)
    _il = [0] * _ds_nn
    log.info(f"LUD diagnoses:")
    for _i, _val in _il_dict.items():
        _ds = _ds_keys[_i]
        log.info(f"{_ds}: {_val}")
        _il[_i] = _val

    _ol_dict = get_output_layer_ce(_p_id, _ds_dict)
    _ol = [0] * _ds_nn
    log.info(f"DN diagnoses:")
    for _i, _val in _ol_dict.items():
        _ds = _ds_keys[_i]
        log.info(f"{_ds}: {_val}")
        _ol[_i] = 1 if _val > 0 else 0

    _dev_name = init_device()

    if os.path.isfile(MODEL_PATH):
        w_model = load_k_model(MODEL_PATH)
        log.info(f"Model was loaded from the {MODEL_PATH}")
    else:
        log.error(f"Keras model file {MODEL_PATH} was not found")
        return None

    _input = np.array([_il], dtype=np.float32)
    _r_list = w_model.predict(_input)

    _nn = len(_r_list[0])
    log.info("Prediction:")
    _r_ds = []
    for _i in range(_nn):
        if _r_list[0][_i] > 0.5:
            _ds = _ds_keys[_i]
            _r_ds.append(_ds)
            log.info(f"{_ds}")

    return _r_ds


if __name__ == "__main__":
    import sys

    n_args = len(sys.argv)
    p_id = int(sys.argv[1]) if n_args > 1 else 896652

    main(p_id)

    sys.exit(0)
