#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2017/10/24 17:48
# created by Jinx

import torch
import torch.nn as nn
from torch.nn import Conv2d,MaxPool2d
from torch.nn.functional import leaky_relu

class Res_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Res_block,self).__init__()
        self
