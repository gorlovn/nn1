#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 09:18 2024

@author: gnv
"""

import os
import redis

import logging
from logging.handlers import RotatingFileHandler

CWD = os.getcwd()
LOG_PATH = os.path.join(CWD, 'log')
LOG_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s %(message)s')  # include timestamp

REDIS_ADDRESS = os.environ.get('REDIS_ADDRESS', 'localhost:6379')
r_arr = REDIS_ADDRESS.split(':')
REDIS_HOST = r_arr[0]
REDIS_PORT = int(r_arr[1])

# соединение с сервером REDIS
REDIS = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)


def setup_logger(logger_name, log_file, log_path=LOG_PATH, level=logging.INFO, console_out=False):
    """
    Function setup as many loggers as you want
    https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
    """

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    l_file = os.path.join(log_path, log_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    handler = RotatingFileHandler(l_file, maxBytes=100000, backupCount=5)
    handler.setFormatter(LOG_FORMATTER)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if console_out:
        console = logging.StreamHandler()
        console.setLevel(level)
        logger.addHandler(console)

    return logger


if __name__ == "__main__":
    log = setup_logger('', '_helpers.out', console_out=True)
else:
    log = logging.getLogger(__name__)

