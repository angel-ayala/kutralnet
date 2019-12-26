import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.mobilenet import ConvBNReLU

class InvertedResidual(nn.Module):
    """
    Taked from the PyTorch repository
    https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class KutralNetMobile(nn.Module): # test 1
    def __init__(self, classes, initial_filters=32):
        super(KutralNetMobile, self).__init__()
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=initial_filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        expansion = 4

        n_filters = initial_filters * 2
        self.block1 = InvertedResidual(initial_filters, n_filters, stride=2, expand_ratio=expansion)

        initial_filters = n_filters
        n_filters = initial_filters * 2
        self.block2 = InvertedResidual(initial_filters, n_filters, stride=2, expand_ratio=expansion)

        initial_filters = n_filters
        n_filters = initial_filters // 2
        self.block3 = InvertedResidual(initial_filters, n_filters, stride=1, expand_ratio=expansion)

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=n_filters)
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
            print('block2.size()', x.size())
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
    model = KutralNetMobile(2)
    print(model)

    from thop import profile, count_hooks, clever_format

    input = torch.randn(1, 3, 84, 84)
    flops, params = profile(model, verbose=False,
            inputs=(input, ),
            custom_ops={torch.nn.Dropout2d: None})
    flops, params = clever_format([flops, params], "%.3f")
    print('Lite model 84x84 params, flops', params, flops)
