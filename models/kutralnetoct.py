"""
transform_compose = transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=85)

Training complete in 3m 56s
Best accuracy on epoch 96: 0.873866
Accuracy of the network on the test images: 81.54%
"""
import torch.nn as nn
import torch.nn.functional as F
from .octave import OctConvBlock
from .octave import _BatchNorm2d
from .octave import _LeakyReLU
from .octave import _MaxPool2d

class KutralNetOctBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, alpha=(0.5, 0.5), padding=1, groups=False, bias=False):
        super(KutralNetOctBlock, self).__init__()
        batch_norm = _BatchNorm2d(out_ch, alpha=alpha)
        activation = _LeakyReLU(inplace=True)
        self.octblock = OctConvBlock(in_channels=in_ch, out_channels=out_ch,
                    kernel_size=kernel_size, stride=stride, alpha=alpha, padding=padding,
                    groups=groups, bias=bias, activation=activation, batch_norm=batch_norm)

        self.pool = _MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.octblock(x)
        x = self.pool(x)
        return x

class KutralNetOct(nn.Module): # test 1
    def __init__(self, classes, initial_filters=32, groups=True):
        super(KutralNetOct, self).__init__()
        oct_alpha = [
            (0., 0.5),
            (0.5, 0.5),
            (0.5, 0.),
        ]

        self.block1 = KutralNetOctBlock(in_ch=3, out_ch=initial_filters, kernel_size=3, stride=1, padding=1,
                        alpha=oct_alpha[0], bias=False)

        n_filters = initial_filters * 2
        self.block2 = KutralNetOctBlock(in_ch=initial_filters, out_ch=n_filters, kernel_size=3, stride=1, padding=1,
                        alpha=oct_alpha[1], groups=groups, bias=False)

        initial_filters = n_filters
        n_filters = initial_filters * 2
        self.block3 = KutralNetOctBlock(in_ch=initial_filters, out_ch=n_filters, kernel_size=3, stride=1, padding=1,
                        alpha=oct_alpha[2], groups=groups, bias=False)

        initial_filters = n_filters
        n_filters = initial_filters // 2
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=initial_filters, out_channels=n_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters)
        )

        self.down_sample = nn.Sequential(
            _MaxPool2d(kernel_size=2, stride=2, padding=0),
            _BatchNorm2d(num_features=n_filters),
            OctConvBlock(in_channels=n_filters, out_channels=n_filters,
                        kernel_size=1, stride=1, alpha=oct_alpha[2], padding=0,
                        groups=groups, bias=False)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_filters, classes)

        self._init_params()

    def forward(self, x):
        debug = False
        if debug:
            print('x.size()', x.size())
        x = self.block1(x)
        if debug:
            print('block1.size()', x.size())
        shortcut = self.block2(x)
        if debug:
            print('block2.size()', x.size())
        x = self.block3(shortcut)
        if debug:
            print('block3.size()', x.size())
        x = self.block4(x)
        if debug:
            print('block4.size()', x.size())
        x += self.down_sample(shortcut)
        # x += shortcut
        # global average pooling
        x = self.global_pool(F.leaky_relu(x))
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
