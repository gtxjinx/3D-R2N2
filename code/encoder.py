#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2017/10/24 17:48
# created by Jinx

from code.net_config import config
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
from torch.nn.functional import leaky_relu, max_pool2d
from torch.autograd import Variable


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, slope=0.1, pooling=True):
        super(Res_block, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, (3, 3), padding=1)
        self.res_conv = Conv2d(in_channels, out_channels, (1, 1))
        self.slope = slope
        self.pooling = pooling

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = leaky_relu(out, negative_slope=self.slope)
        out = self.conv2(out)
        out = leaky_relu(out, negative_slope=self.slope)
        res = self.res_conv(res)
        out = out + res
        if self.pooling:
            out = max_pool2d(out, (2, 2), stride=2, padding=0)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2d(3, 96, (7, 7), padding=3)
        self.conv2 = Conv2d(96, 96, (3, 3), padding=1)
        self.res_conv = Conv2d(3, 96, (1, 1))
        self.conv3 = Conv2d(256, 256, (3, 3), padding=1)
        self.conv4 = Conv2d(256, 256, (3, 3), padding=1)
        self.res_block1 = Res_block(96, 128)
        self.res_block2 = Res_block(128, 256)
        self.res_block3 = Res_block(256, 256)
        self.res_block4 = Res_block(256, 256, pooling=False)
        self.fc = Linear(256 * 4 * 4, 1024)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = leaky_relu(out, 0.1)
        out = self.conv2(out)
        out = leaky_relu(out, 0.1)
        res = self.res_conv(res)
        out = out + res
        out = max_pool2d(out, (2, 2), stride=2, padding=1)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.conv3(out)
        out = leaky_relu(out, 0.1)
        out = self.conv4(out)
        out = leaky_relu(out, 0.1)
        out = max_pool2d(out, (2, 2), stride=2)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = out.view(-1, 256 * 4 * 4)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    encoder = Encoder()
    print(len(list(encoder.parameters())))
    a = Variable(torch.rand(36, 3, 127, 127))
    a = encoder(a)
    print(a)
