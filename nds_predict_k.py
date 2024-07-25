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
import time
import numpy as np
from tensorflow.keras.models import load_model

import logging

from slib.slib_config import JSONRPC_DBMIS_API_URI
from slib.m_utils import dbmis_api_execute

from settings import MODEL_PATH
from settings import RAM_MODEL_PATH
from helpers import setup_logger

from nds import get_diagnosis_dict
from nds_train_k import init_device

if __name__ == "__main__":
    log = setup_logger('', '_nds_predict_k.out', console_out=True)
else:
    log = logging.getLogger(__name__)

# Список диагнозов пациента из ЛУД
SQLT_GET_DD_DIAGNOSES = "SELECT TRIM(diagnosis_id_fk) FROM disease_diagnosis WHERE people_id_fk = ?;"

# Список диагнозов пациента из ДН
SQLT_GET_CE_DIAGNOSES = ("SELECT TRIM(diagnosis_id_fk) FROM clinical_examinations WHERE people_id_fk = ? "
                         "and diagnosis_id_fk is not null;")


def get_input_layer_dd(_p_id, _ds_dict) -> dict:
    """
    Сформировать input layer для пациента с ИД _p_id
    на основании диагнозов ЛУД (disease_diagnosis)

    Возвращаем словарь с ненулевыми значениями входного слоя
    """

    _rs = dbmis_api_execute(SQLT_GET_DD_DIAGNOSES, [_p_id], True)

    if type(_rs) is list:
        _il = {}
        for _r in _rs:
            _ds = _r[0]
            _i = _ds_dict.get(_ds)
            if _i in _il:
                _il[_i] += 1
            else:
                _il[_i] = 1

        return _il

    log.warning(f"Wrong result getting dd input layer for patient {_p_id} over {JSONRPC_DBMIS_API_URI}: {_rs}")
    return {}


def get_output_layer_ce(_p_id, _ds_dict) -> dict:
    """
    Сформировать output layer для пациента с ИД _p_id
    на основании диагнозов таблицы ДН (clinincal_examinations)

    Возвращаем словарь с ненулевыми значениями входного слоя
    """

    _rs = dbmis_api_execute(SQLT_GET_CE_DIAGNOSES, [_p_id], True)

    if type(_rs) is list:
        _ol = {}
        for _r in _rs:
            _ds = _r[0]
            _i = _ds_dict.get(_ds)
            if _i in _ol:
                _ol[_i] += 1
            else:
                _ol[_i] = 1

        return _ol

    log.warning(f"Wrong result getting ce output layer for patient {_p_id} over {JSONRPC_DBMIS_API_URI}: {_rs}")
    return {}


def load_k_model(_ram_model_path=RAM_MODEL_PATH, _model_path=MODEL_PATH):

    if os.path.isfile(_ram_model_path):
        _m_path = _ram_model_path
    elif os.path.isfile(_model_path):
        _m_path = _model_path
    else:
        log.info(f"ram_model_path: {_ram_model_path}")
        log.info(f"model_path: {_model_path}")
        log.error("Saved model was not found")
        return None

    log.info(f"++++++++ Loading keras model from {_m_path}")
    _start = time.time()
    _model = load_model(_m_path)
    _model.make_predict_function()
    _model.summary()  # Optional: Include this line to display the model structure when the model is reloaded
    _dur = time.time() - _start
    log. info(f"Loaded in {_dur:.2f} sec")

    return _model


def main(_p_id, _model):

    _dev_name = init_device()

    log.info(f"******** Predict DN diagnosis list for the patient {_p_id} ********")
    _start = time.time()

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

    _input = np.array([_il], dtype=np.float32)
    _r_list = _model.predict(_input)

    _nn = len(_r_list[0])
    log.info("Prediction:")
    _r_ds = []
    for _i in range(_nn):
        if _r_list[0][_i] > 0.5:
            _ds = _ds_keys[_i]
            _r_ds.append(_ds)
            log.info(f"{_ds}")

    _dur = time.time() - _start
    log.info(f"Duration {_dur:.2f} sec")
    return _r_ds


if __name__ == "__main__":
    import sys

    w_model = load_k_model()
    if w_model is None:
        sys.exit(1)

    n_args = len(sys.argv)
    if n_args <= 1:
        main(896652, w_model)
    else:
        for i_arg in range(1, n_args):
            p_id = int(sys.argv[i_arg])
            main(p_id, w_model)

    sys.exit(0)
