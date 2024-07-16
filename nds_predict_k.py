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
from tensorflow.keras.models import load_model

import logging

from helpers import setup_logger

from nds import get_diagnosis_dict
from nds_model_k import model
from nds_train_k import MODEL_PATH
from nds_train_k import init_device

from nds_predict import get_input_layer_dd, get_output_layer_ce

if __name__ == "__main__":
    log = setup_logger('', '_nds_predict_k.out', console_out=True)
else:
    log = logging.getLogger(__name__)


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
        w_model = load_model(MODEL_PATH)
        log.info(f"Model was loaded from the {MODEL_PATH}")
    else:
        log.error(f"Model data file {MODEL_PATH} was not found")
        return None

    _input = np.array(_il, dtype=np.float32)
    _r_list = w_model.predict(_input)

    _nn = len(_r_list)
    log.info("Prediction:")
    _r_ds = []
    for _i in range(_nn):
        if _r_list[_i] > 0.5:
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

