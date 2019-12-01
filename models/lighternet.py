"""
224x224
Training complete in 118m 10s
Best val Acc: 0.941015

68x68
Training complete in 8m 31s
Best val Acc: 0.943759
Test Acc: 0.9357
"""

import torch.nn as nn
import torch.nn.functional as F
from octconv import OctConv2d

def _conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def _conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1,
                  stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class _BatchNorm2d(nn.Module):
    def __init__(self, num_features, alpha=(0.25, 0.25), eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm2d, self).__init__()
        alpha_in = alpha[1]
        hf_ch = int(num_features * (1 - alpha_in))
        lf_ch = num_features - hf_ch
        self.bnh = nn.BatchNorm2d(hf_ch)
        self.bnl = nn.BatchNorm2d(lf_ch)

    def forward(self, x):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)

class _ReLU6(nn.ReLU6):
    def forward(self, x):
        hf, lf = x
        hf = super(_ReLU6, self).forward(hf)
        lf = super(_ReLU6, self).forward(lf)
        return hf, lf

class _AvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        hf, lf = x
        hf = super(_AvgPool2d, self).forward(hf)
        lf = super(_AvgPool2d, self).forward(lf)
        return hf, lf

def _octconv_bn_relu(inp, oup, kernel=3, stride=1, alpha=(0.125, 0.125), padding=0, activation=True):
    mods = []
    mods.append(OctConv2d(in_channels=inp, out_channels=oup, kernel_size=kernel,
              stride=stride, alpha=alpha, padding=padding))
    mods.append(_BatchNorm2d(oup, alpha=alpha))

    if activation:
        mods.append(_ReLU6(inplace=True))

    return nn.Sequential(*mods)

def _octconv_final_bn(inp, oup, kernel=3, stride=1, alpha=(0.125, 0.), padding=0):
    return nn.Sequential(
        OctConv2d(in_channels=inp, out_channels=oup, kernel_size=kernel,
                  stride=stride, alpha=alpha, padding=padding),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class LighterNet(nn.Module):
    def __init__(self, classes):
        super(LighterNet, self).__init__()
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

        alphas = [
            (0., .25),
            (.25, .25),
            (.25, 0.)
        ]

        self.firstOctave = _octconv_bn_relu(16, 32, kernel=3, stride=1, alpha=alphas[0])
        self.avg_pool = _AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.middleOctave = _octconv_bn_relu(32, 64, kernel=3, stride=1, alpha=alphas[1])
        self.avg_pool2 = _AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.lastOctave = _octconv_final_bn(64, 64, kernel=3, stride=1, alpha=alphas[2])
        # global average pooling
        # self.last_pool = nn.AvgPool2d(kernel_size=24)
        self.classifier = nn.Linear(64, classes)
        self._init_params()

    def forward(self, x):
        x = self.firstBlock(x)
        x = self.firstOctave(x)
        x = self.avg_pool(x)
        x = self.middleOctave(x)
        x = self.avg_pool2(x)
        x = self.lastOctave(x)
        # x = self.last_pool(x)
        # print('x.size()', x.size(2))
        x = F.avg_pool2d(x, x.size(2))
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)#, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
