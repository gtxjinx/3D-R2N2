#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2017/10/25 17:15
# created by Jinx

from code.net_config import config
import torch
import torch.nn as nn
from torch.nn import Conv3d, Linear
from torch.nn.functional import leaky_relu, max_pool2d
from torch.autograd import Variable


class FCConv_block(nn.Module):
    def __init__(self):
        super(FCConv_block, self).__init__()
        # inchannels????
        self.conv = Conv3d(256, 128, (3, 3, 3), padding=1)
        self.fc = Linear(1024, 128 * 4 * 4 * 4)

    def forward(self, x, hidden):
        out1 = self.fc(x)
        out1.view(128, 4, 4, 4)
        out2 = self.conv(hidden)

        return out1 + out2


class Recurrence(nn.Module):
    pass
