#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Тренировка модели

Created on Wen Jul 03 15:58 2024

@author: gnv
"""
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import logging

from slib.utils import setup_logger

from nds import construct_training_data_set
from nds_model import model

if __name__ == "__main__":
    log = setup_logger('', '_nds_train.out', console_out=True)
else:
    log = logging.getLogger(__name__)


def train(_nn=1000, _i_start=0, _n_epochs=100, _batch_size=10):

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)

    log.info(f"******** NDS model training on {device_name} ********")
    _xt, _yt = construct_training_data_set(_nn, _i_start)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _epoch in tqdm(range(_n_epochs)):
        loss = None
        for _i in range(0, _nn, _batch_size):
            _x_batch = _xt[_i:_i++_batch_size]
            _y_pred = model(_x_batch)
            _y_batch = _yt[_i:_i++_batch_size]
            loss = loss_fn(_y_pred, _y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Finished epoch {_epoch}, latest loss {loss}')


if __name__ == "__main__":

    train()
