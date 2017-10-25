#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2017/10/25 15:40
# created by Jinx

import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d,Linear
from torch.nn.functional import leaky_relu,adaptive_avg_pool2d,max_pool2d
from torch.autograd import Variable

a=Variable(torch.ones(2,50))
fc=Linear(50,20)
print(fc(a))
# for i in range(7):
#     a[:,i,:]=a[:,i,:]*(i+1)
# print(a)
# b=max_pool2d(a,(2,2),padding=1)
# print(b)
# c=b.mean()
#
# c.backward()
# print(a.grad)

