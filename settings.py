#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

BASE_PATH = Path(__file__).resolve().parent
_dot_env_path = os.path.join(BASE_PATH, '.env')
if os.path.isfile(_dot_env_path):
    load_dotenv(dotenv_path=_dot_env_path)

# Проверка на запуск в тестовой среде
_tdev = os.environ.get('TESTDEV')
if _tdev is None:
    # По умолчанию тестовый контур
    TESTDEV = True
elif _tdev[0] in ('1', 'T', 't'):
    # Тестировать на dev контуре
    TESTDEV = True
else:
    TESTDEV = False

_nn1_settings_file = os.getenv('NN1_SETTINGS_CFG', 'nn1_settings.yaml')
_nn1_settings_path = os.path.join(BASE_PATH, _nn1_settings_file)
if os.path.isfile(_nn1_settings_path):
    with open(_nn1_settings_path, "r") as f:
        NN1_SETTINGS = yaml.safe_load(f)
else:
    NN1_SETTINGS = {}

DATA_PATH = os.path.join(BASE_PATH, 'data')
MODEL_FILE = NN1_SETTINGS.get('model_file', 'nds_model.keras')
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

PYRO_EXPOSE_ADDRESS = NN1_SETTINGS.get('pyro_expose_address', '127.0.0.1')


if __name__ == "__main__":

    print(f"BASE_PATH: {BASE_PATH}")
    model_found = os.path.isfile(MODEL_PATH)
    print(f"MODEL_PATH: {MODEL_PATH}. Found: {model_found}")
