#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2017/10/25 17:15
# created by Jinx

from code.net_config import config
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
from torch.nn.functional import leaky_relu, max_pool2d
from torch.autograd import Variable


class Recurrence(nn.Module):
    pass
