#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель
"""
import os
import torch.nn as nn

DS_NN = int(os.environ.get('DS_NN', '14646'))

NN1 = 256
NN2 = 128
NN3 = 512

model = nn.Sequential(
    nn.Linear(DS_NN, NN1),
    nn.ReLU(),
    nn.Linear(NN1, NN2),
    nn.ReLU(),
    nn.Linear(NN2, NN3),
    nn.ReLU(),
    nn.Linear(NN3, DS_NN),
    nn.ReLU(),
    nn.Sigmoid()
)



def test_model():
    print(model)


if __name__ == "__main__":

    test_model()
