#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Тренировка модели

Created on Wen Jul 03 15:58 2024

@author: gnv
"""
import gc
import time

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


def clear_cuda_memory():
    """
    https://stackoverflow.com/questions/55322434/how-to-clear-cuda-memory-in-pytorch
    :return:
    """
    torch.cuda.empty_cache()
    gc.collect()
    log.info("CUDA memory has been cleared")


def train(_nn=1000, _i_start=0, _n_epochs=100, _batch_size=10):

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith('cuda'):
        clear_cuda_memory()

    device = torch.device(device_name)
    model.to(device)

    log.info(f"******** NDS model training on {device_name} ********")
    _xt, _yt = construct_training_data_set(_nn, _i_start)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    _start = time.time()
    for _epoch in range(_n_epochs):
        model.train()
        for _i in range(0, _nn, _batch_size):
            optimizer.zero_grad()

            _x_batch = _xt[_i:_i++_batch_size]
            _y_pred = model(_x_batch.to(device))
            _y_batch = _yt[_i:_i++_batch_size].to(device)

            loss = loss_fn(_y_pred, _y_batch)

            loss.backward()
            optimizer.step()

        # Validation loop (optional)
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        acc = 0
        count = 0
        with torch.no_grad():  # No need to track gradients for validation
            for _i in range(_nn):
                _predict = model(_xt[_i].to(device))
                _target = _yt[_i].to(device)
                val_loss += loss_fn(_predict, _target).item()
                _cmp = (torch.round(_predict) == _target)
                acc += _cmp.float().sum()
                count += len(_target)

        avg_val_loss = val_loss / _nn
        _dur = (time.time() - _start)  # sec
        acc /= count
        print(f"{_dur:.2f}: Epoch {_epoch + 1}/{_n_epochs}, Validation Loss: {avg_val_loss:.4f}, "
              f"Model accuracy: {acc*100:.2f}")


if __name__ == "__main__":

    train()
