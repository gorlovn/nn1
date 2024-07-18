#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Тест сохранения данных модели в redis

Created on Thu Jul 18 10:17 2024

@author: gnv
"""
import os
import redis

import logging

from helpers import setup_logger
from helpers import REDIS
from settings import MODEL_PATH, MODEL_JSON_PATH, MODEL_WEIGHTS_PATH
from settings import MODEL_ARCHITECTURE_KEY,  MODEL_WEIGHTS_KEY, REDIS_EX

if __name__ == "__main__":
    log = setup_logger('', '_stote_to_reddis.out', console_out=True)
else:
    log = logging.getLogger(__name__)


def read_weights_and_store_to_redis(_weights_path=MODEL_WEIGHTS_PATH):

    log.info(f"Reading the model weights from the {_weights_path}")
    with open(_weights_path, 'rb') as f:
        model_h5 = f.read()

    # Store the model weights in Redis
    log.info(f"Storing the model weights to the redis {MODEL_WEIGHTS_KEY}")
    REDIS.set(MODEL_WEIGHTS_KEY, model_h5, ex=REDIS_EX)

    log.info("Done")


def test1():

    read_weights_and_store_to_redis()


if __name__ == "__main__":

    test1()
