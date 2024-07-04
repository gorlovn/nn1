#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Работа с подготовленными данными по диагнозам

Created on Wen Jul 03 13:22 2024

@author: gnv
"""
import os
import json
import pickle
import time
from tqdm import tqdm

import logging

from helpers import setup_logger
from helpers import REDIS

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
DS_DATA_FILE = 'diagnosis_dict.json'
DS_DATA_PATH = os.path.join(DATA_PATH, DS_DATA_FILE)
DATASET_FILE = 'dataset_v1.dat'

REDIS_TIME_TO_LIVE = int(os.environ.get('REDIS_DS_TIME_TO_LIVE', '3600'))
REDIS_DS_KEY = os.environ.get('REDIS_DS_KEY', 'diagnosis_dict')

# Кол-во диагнозов
DS_NN = None

if __name__ == "__main__":
    log = setup_logger('', '_ds.out', console_out=True)
else:
    log = logging.getLogger(__name__)


def get_diagnosis_dict():
    global DS_NN

    _diagnosis_dict_p = REDIS.get(REDIS_DS_KEY)
    if _diagnosis_dict_p:
        _diagnosis_dict = pickle.loads(_diagnosis_dict_p)

        DS_NN = len(_diagnosis_dict)
        log.info(f"We have got {DS_NN} diagnosis codes from the Redis")

        return _diagnosis_dict

    if os.path.isfile(DS_DATA_PATH):
        with open(DS_DATA_PATH, 'r', encoding='utf-8') as _f:
            try:
                _diagnosis_dict = json.load(_f)
            except Exception as _e:
                _diagnosis_dict = None
                log.warning(f"Error getting diagnosis dict from the {DS_DATA_PATH}: {_e}")

        if type(_diagnosis_dict) is dict:
            DS_NN = len(_diagnosis_dict)
            log.info(f"We have got {DS_NN} diagnosis codes from the {DS_DATA_PATH}")

            _diagnosis_dict_p = pickle.dumps(_diagnosis_dict)
            REDIS.set(REDIS_DS_KEY, _diagnosis_dict_p, ex=REDIS_TIME_TO_LIVE)
            return _diagnosis_dict

    log.error("Diagnosis dict data was not found")
    return None


def load_dataset(_dataset_file=DATASET_FILE, _data_path=DATA_PATH):

    _dataset_path = os.path.join(_data_path, _dataset_file)
    if os.path.isfile(_dataset_path):
        with open(_dataset_path, 'rb') as handle:
            _dataset = pickle.load(handle)

        if type(_dataset) is dict:
            _p_ids = list(_dataset.keys())
            _nn = len(_p_ids)
            log.info(f"We have got {_nn} patients data from the {_dataset_path}")
        else:
            log.error(f"Wrong type of data in the {_dataset_path}: {type(_dataset)}")
    else:
        log.error(f"File {_dataset_path} was not found")
        _dataset = None

    return _dataset


def construct_layers(_p_id, _dataset):
    """

    Конструируем слои для _i-го пациента в наборе данных
    :param _p_id:
    :param _dataset:
    :return:
    """

    global DS_NN

    _data = _dataset.get(_p_id)
    if type(_data) is not list or len(_data) != 2:
        log.error(f"Can't get layers of the {_p_id} patient")
        log.error(f"Patients' data: {_data}")
        return None

    _il_dict = _data[0]
    _il = [0] * DS_NN
    for _i, _val in _il_dict.items():
        _il[_i] = _val

    _ol_dict = _data[1]
    _ol = [0] * DS_NN
    for _i, _val in _ol_dict.items():
        _ol[_i] = 1 if _val > 0 else 0

    return _il, _ol


def construct_training_data_set(_nn=500, _i_start=0):
    """
    Сформировать набор данных для тренировки сети
    :param _nn: количество слоев
    :param _i_start: с какого по счету пациента начинать

    :return:
    """
    import torch

    _start = time.time()
    log.info("++++++++ Construct training data +++++++")
    log.info("+++++ get_diagnosis_dict")
    _ds_dict = get_diagnosis_dict()
    if _ds_dict is None:
        return None, None

    log.info("+++++ load_dataset")
    _dataset = load_dataset()

    _p_ids = list(_dataset.keys())
    _nn_p_ids = len(_p_ids)

    _x = []
    _y = []
    _i_finish = _i_start + _nn
    log.info(f"{_i_start}:{_i_finish}")
    for _i in tqdm(range(_i_start, _i_finish)):
        if _i > _nn_p_ids:
            break

        _p_id = _p_ids[_i]
        _il, _ol = construct_layers(_p_id, _dataset)
        _x.append(_il)
        _y.append(_ol)

    _xt = torch.tensor(_x, dtype=torch.float32)
    _yt = torch.tensor(_y, dtype=torch.float32)

    _dur = (time.time() - _start) * 1000
    log.info(f"Duration: {_dur:.2f} ms")
    return _xt, _yt


def test_construct_layers(_i, _p_ids, _dataset):
    log.info("+++++ Construct layers")
    log.info(f"Patient's index: {_i}")
    _p_id = _p_ids[_i]
    log.info(f"Patient: {_p_id}")
    _il, _ol = construct_layers(_p_id, _dataset)

    if type(_il) is list:
        _nn_il = len(_il)
        log.info(f"Input layer length: {_nn_il}")
    else:
        log.error(f"Wrong input layer: {_il}")

    if type(_ol) is list:
        _nn_ol = len(_ol)
        log.info(f"Output layer length: {_nn_ol}")
    else:
        log.error(f"Wrong output layer: {_ol}")


def test_module():
    _start = time.time()

    log.info("++++++++ Test Module ++++++++")
    log.info("+++++ get_diagnosis_dict")
    _ds_dict = get_diagnosis_dict()
    log.info("+++++ load_dataset")
    _dataset = load_dataset()

    if type(_dataset) is dict:
        _p_ids = list(_dataset.keys())
        test_construct_layers(7, _p_ids, _dataset)
        test_construct_layers(113, _p_ids, _dataset)

    _tds = construct_training_data_set(10)

    _dur = (time.time() - _start) * 1000
    log.info(f"Duration: {_dur:.2f} ms")


if __name__ == "__main__":
    import sys

    test_module()

    sys.exit(0)
