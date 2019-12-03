"""
Training complete in _m _s
Best val Acc: X
Test acc: Y
"""
import torch.nn as nn
import torch.nn.functional as F
from octconv import OctConv2d
from .octave import _BatchNorm2d
from .octave import _AvgPool2d
from .octave import _SELU
from .octave import _octconv_bn

def octconv_bn(in_channels,
                out_channels,
                kernel=3,
                stride=1,
                alpha=(.25, .25),
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                activation=True):
    return _octconv_bn(in_channels, out_channels, kernel=kernel,
                    stride=stride, alpha=alpha, padding=padding,
                    dilation=dilation, groups=groups, bias=bias,
                    activation=activation, act_layer=_SELU)

class KutralNet(nn.Module):
    def __init__(self, classes):
        super(KutralNet, self).__init__()
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

        alphas = [
            (0., .25),
            (.25, .25),
            (.25, 0.)
        ]

        self.firstOctave = octconv_bn(64, 128, kernel=3, stride=1, alpha=alphas[0])
        self.avg_pool = _AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.middleOctave = octconv_bn(128, 32, kernel=3, stride=1, alpha=alphas[1], groups=4)
        self.avg_pool2 = _AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.lastOctave = octconv_bn(32, 64, kernel=3, stride=1, alpha=alphas[2], padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, classes)
        )
        self._init_params()

    def forward(self, x):
        debug = False
        if debug:
            print('x.size()', x.size(2))
        x = self.firstBlock(x)
        if debug:
            print('x.size()', x[0].size(2), x[1].size(2))
        x = self.firstOctave(x)
        if debug:
            print('x.size()', x[0].size(2), x[1].size(2))
        x = self.avg_pool(x)
        if debug:
            print('x.size()', x[0].size(2), x[1].size(2))
        x = self.middleOctave(x)
        if debug:
            print('x.size()', x[0].size(2), x[1].size(2))
        x = self.avg_pool2(x)
        if debug:
            print('x.size()', x[0].size(2), x[1].size(2))
        x = self.lastOctave(x)
        # x = self.last_pool(x)
        if debug:
            print('x.size()', x.size(2))
        # global average pooling
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)#, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
