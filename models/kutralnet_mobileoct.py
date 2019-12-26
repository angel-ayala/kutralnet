import torch
from torch import nn
from torch.nn import functional as F
from octconv import OctConv2d
from .octave import OctConvBlock
from .octave import _BatchNorm2d
from .octave import _LeakyReLU
from .octave import _MaxPool2d

class InvertedResidualOct(nn.Module):
    """
    Taked from the PyTorch repository and modified to work with octave convolution
    https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
    """
    def __init__(self, inp, oup, stride, expand_ratio, alpha=(0.5, 0.5)):
        super(InvertedResidualOct, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        is_initial = alpha[0] == 0
        is_final = alpha[1] == 0

        alpha_1 = alpha_2 = alpha_3 = alpha

        if is_initial:
            alpha_2 = (alpha[1], alpha[1])
            alpha_3 = (alpha[1], alpha[1])

        if is_final:
            alpha_1 = (alpha[0], alpha[0])
            alpha_3 = (alpha[1], alpha[1])

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(OctConvBlock(in_channels=inp, out_channels=hidden_dim,
                        kernel_size=1, stride=1, alpha=alpha_1, padding=0,
                        activation=_LeakyReLU(inplace=True), batch_norm=_BatchNorm2d(hidden_dim, alpha=alpha_1)))
            alpha_1 = alpha_2 if not is_final else alpha_1

        layers.extend([
            # dw
            OctConvBlock(in_channels=hidden_dim, out_channels=hidden_dim,
                    kernel_size=3, stride=stride, alpha=alpha_1, padding=1,
                    groups=True, activation=_LeakyReLU(inplace=True), batch_norm=_BatchNorm2d(hidden_dim, alpha=alpha_1)),
            # pw-linear
            OctConv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, alpha=alpha_2),
            _BatchNorm2d(oup, alpha=alpha_3),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class KutralNetMobileOct(nn.Module): # test 1
    def __init__(self, classes, initial_filters=32):
        super(KutralNetMobileOct, self).__init__()
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=initial_filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        expansion = 4
        oct_alpha = [
            (0., 0.5),
            (0.5, 0.5),
            (0.5, 0.),
            # (0., 0.25),
            # (0.25, 0.25),
            # (0.25, 0.),
        ]

        n_filters = initial_filters * 2
        self.block1 = InvertedResidualOct(initial_filters, n_filters, stride=2, expand_ratio=expansion, alpha=oct_alpha[0])

        initial_filters = n_filters
        n_filters = initial_filters * 2
        self.block2 = InvertedResidualOct(initial_filters, n_filters, stride=2, expand_ratio=expansion, alpha=oct_alpha[1])

        initial_filters = n_filters
        n_filters = initial_filters // 2
        self.block3 = InvertedResidualOct(initial_filters, n_filters, stride=1, expand_ratio=expansion, alpha=oct_alpha[2])

        self.down_sample = nn.Sequential(
            _MaxPool2d(kernel_size=2, stride=2, padding=0),
            _BatchNorm2d(num_features=n_filters, alpha=oct_alpha[1]),
            OctConvBlock(in_channels=n_filters, out_channels=n_filters,
                        kernel_size=1, stride=1, alpha=oct_alpha[2], padding=0,
                        groups=True, bias=False)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_filters, classes)

        self._init_params()

    def forward(self, x):
        debug = False
        if debug:
            print('x.size()', x.size())
        x = self.firstBlock(x)
        if debug:
            print('firstBlock.size()', x.size())
        shortcut = self.block1(x)
        if debug:
            print('block1.size()', x.size())
        x = self.block2(shortcut)
        if debug:
            print('block2.size()', x[0].size(), x[1].size())
        x = self.block3(x)
        if debug:
            print('block3.size()', x.size())
        # ds = self.down_sample(shortcut)
        # if debug:
        #     print('down_sample.size()', ds.size())
        x += self.down_sample(shortcut)
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

if __name__ == '__main__':
    model = KutralNetMobileOct(2)
    print(model)

    from thop import profile, count_hooks, clever_format

    input = torch.randn(1, 3, 84, 84)
    flops, params = profile(model, verbose=False,
            inputs=(input, ),
            custom_ops={torch.nn.Dropout2d: None})
    flops, params = clever_format([flops, params], "%.3f")
    print('Lite model 84x84 params, flops', params, flops)
