#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Для заданного пациента
сформировать предполагаемые диагнозы ДН
используя сохраненную модель

Created on Thu Jul 04 10:42 2024

@author: gnv
"""
import os
import torch

import logging

from slib.slib_config import JSONRPC_DBMIS_API_URI
from slib.m_utils import dbmis_api_execute

from helpers import setup_logger

from nds import get_diagnosis_dict
from nds_model import model
from nds_train import MODEL_PATH
from nds_train import clear_cuda_memory

if __name__ == "__main__":
    log = setup_logger('', '_nds_predict.out', console_out=True)
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

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith('cuda'):
        clear_cuda_memory()

    device = torch.device(device_name)
    model.to(device)

    log.info(f"+++++ Load NDS model to the {device_name}")

    if os.path.isfile(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except torch.cuda.OutOfMemoryError:
            log.warning("CUDA out of memory. Use CPU.")
            device_name = 'cpu'
            device = torch.device(device_name)
            model.to(device)
            model.load_state_dict(torch.load(MODEL_PATH))

        log.info(f"Model was loaded from the {MODEL_PATH}")
    else:
        log.error(f"Model data file {MODEL_PATH} was not found")
        return None

    _input = torch.tensor(_il, dtype=torch.float32)
    _output = model(_input.to(device))
    _result = torch.round(_output)
    _r_list = _result.tolist()
    _nn = len(_r_list)
    log.info("Prediction:")
    _r_ds = []
    for _i in range(_nn):
        if _r_list[_i] > 0:
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




