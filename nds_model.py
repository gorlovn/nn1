#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель
"""
import os
import torch.nn as nn

DS_NN = int(os.environ.get('DS_NN', '14646'))

NN1 = 1024
NN2 = 512
NN3 = 2048

model0 = nn.Sequential(
    nn.Linear(DS_NN, DS_NN),
    nn.ReLU(),
    nn.Sigmoid()
)


class IdentityModel(nn.Module):
    def __init__(self, input_size):
        super(IdentityModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.linear(x)


model = IdentityModel(DS_NN)


def test_model():
    print(model)


if __name__ == "__main__":

    test_model()
