import torch.nn as nn
import torch.nn.functional as F
from octconv import OctConv2d


def _conv_bn_relu(in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding='same',
                bias=False,
                activation=True):

    if padding == 'same':
        padding = (kernel_size -1) // 2
    elif type(padding) == int:
        padding = padding

    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, bias=bias)
    ]
    layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_channels,
                out_channels,
                stride=1,
                downsample_shortcut=False,
                expansion=4):
        super(Bottleneck, self).__init__()
        final_filters = int(out_channels * expansion)

        self.conv1 = _conv_bn_relu(in_channels, out_channels, kernel_size=1)
        self.conv2 = _conv_bn_relu(out_channels, out_channels, kernel_size=3, stride=stride)
        self.conv3 = _conv_bn_relu(out_channels, final_filters, kernel_size=1, activation=False)

        self.downsample = nn.Conv2d(out_channels, final_filters, kernel_size=1, stride=stride, bias=False) if downsample_shortcut else None


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)
        return x


def _bottleneck_original(ip, filters, strides=(1, 1), downsample_shortcut=False,
                         expansion=4):

    final_filters = int(filters * expansion)

    shortcut = ip

    x = _conv_bn_relu(ip, filters, kernel_size=(1, 1))
    x = _conv_bn_relu(x, filters, kernel_size=(3, 3), strides=strides)
    x = _conv_bn_relu(x, final_filters, kernel_size=(1, 1), activation=False)

    if downsample_shortcut:
        shortcut = _conv_block(shortcut, final_filters, kernel_size=(1, 1),
                               strides=strides)

    x = add([x, shortcut])
    x = ReLU()(x)

    return x

def octconv(in_channels, #ip,
            out_channels, #filters,
            kernel_size=3, #(3, 3),
            stride=1, #(1, 1),
            alpha=(.5, .5),
            padding='same',
            # dilation=None,
            bias=False):
    if padding == 'same':
        padding = (kernel_size -1) // 2
    elif type(padding) == int:
        padding = padding

    return OctConv2d(in_channels,
                 out_channels,
                 kernel_size,
                 stride=stride,
                 padding=padding,
                 alpha=alpha,
                 bias=bias)

class _BatchNorm2d(nn.Module):
    def __init__(self,
                num_features,
                alpha=(0.5, 0.5),
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True):
        super(_BatchNorm2d, self).__init__()

        alpha_out = alpha[1] if type(alpha) in [tuple, list] else alpha
        hf_ch = int(num_features * (1 - alpha_out))
        lf_ch = num_features - hf_ch
        self.bnh = nn.BatchNorm2d(hf_ch)
        self.bnl = nn.BatchNorm2d(lf_ch)

    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            return self.bnh(hf), self.bnl(lf)
        else:
            return self.bnh(x)

class _ReLU(nn.ReLU):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_ReLU, self).forward(hf)
            lf = super(_ReLU, self).forward(lf)
            return hf, lf
        else:
            return super(_ReLU, self).forward(x)

def octconv_bn(in_channels,
                out_channels,
                kernel=3,
                stride=1,
                alpha=(.5, .5),
                padding='same',
                # dilation=None,
                bias=False,
                activation=True):
    mods = []
    mods.append(octconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel,
                stride=stride, alpha=alpha, padding=padding,
                # dilation=None,
                bias=bias))
    mods.append(_BatchNorm2d(out_channels, alpha=alpha))

    if activation:
        mods.append(_ReLU(inplace=True))

    return nn.Sequential(*mods)

def octconv_bn_final(in_channels,
                out_channels,
                kernel=3,
                stride=1,
                alpha=(.5, 0.),
                padding='same',
                # dilation=None,
                bias=False,
                activation=True):
    return nn.Sequential(
        octconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel,
                    stride=stride, alpha=alpha, padding=padding,
                    # dilation=None,
                    bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class OctconvBottleneck(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                alpha=(.5, .5),
                stride=1,
                downsample_shortcut=False,
                first_block=False,
                last_block=False,
                expansion=4):
        super(OctconvBottleneck, self).__init__()
        block1_alpha = block3_alpha = alpha
        final_out_channels = int(out_channels * expansion)

        if first_block:
            block1_alpha = (0., block1_alpha[1])
        else:
            in_channels = int(in_channels * expansion)

        if last_block:
            block3_alpha = (block3_alpha[0], 0.)

        self.conv1 = octconv_bn(in_channels, out_channels, kernel=1, alpha=block1_alpha)
        self.conv2 = octconv_bn(out_channels, out_channels, kernel=3, stride=stride, alpha=alpha)
        self.conv3 = octconv_bn(out_channels, final_out_channels, kernel=1, alpha=block3_alpha,
                                activation=False)

        if first_block:
            block3_alpha = (0., block3_alpha[1])

        self.downsample = octconv_bn(in_channels, final_out_channels, kernel=1, stride=stride,
                        alpha=block3_alpha, activation=False) if downsample_shortcut else None


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # shortcut
        x_h, x_l = x if isinstance(x, tuple) else (x, None)
        identity_h, identity_l = identity if isinstance(identity, tuple) else (identity, None)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = F.relu(x_h)
        x_l = F.relu(x_l) if x_l is not None else None
        x = (x_h, x_l) if x_l is not None else x_h

        return x

class OctaveResNet(nn.Module):
    def __init__(self, layers,
                 classes=1000,
                 alpha=(.5, .5),
                 expansion=1,
                 initial_filters=64,
                 initial_strides=False,
                 **kwargs):

        super(OctaveResNet, self).__init__()

        if type(layers) not in [list, tuple]:
            raise ValueError('`layers` must be a list/tuple of integers. '
                         'Current layers = ', layers)

        # Force convert all layer values to integers
        layers = [int(x) for x in layers]

        if initial_strides:
            initial_strides = 2
        else:
            initial_strides = 1

        first_block_layers = [
            nn.Conv2d(in_channels=3, out_channels=initial_filters, kernel_size=7,
                        stride=initial_strides, padding=3, bias=False),
            nn.BatchNorm2d(num_features=initial_filters),
            nn.ReLU(inplace=True),
        ]

        if initial_strides:
            first_block_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.first_block = nn.Sequential(*first_block_layers)

        num_filters = initial_filters
        num_blocks = len(layers)
        res_blocks = []

        for i in range(num_blocks - 1):
            for j in range(layers[i]):
                if j == 0:
                    strides = 2
                    downsample_shortcut = True
                else:
                    strides = 1
                    downsample_shortcut = False
                # first block has no downsample, no shortcut
                if i == 0 and j == 0:
                    first_block = True
                    strides = 1
                    downsample_shortcut = True
                else:
                    first_block = False

                res_blocks.append(OctconvBottleneck(in_channels=num_filters, out_channels=num_filters,
                                alpha=alpha, stride=strides, downsample_shortcut=downsample_shortcut,
                                first_block=first_block, expansion=expansion))

            # double number of filters per block
            num_filters *= 2

        self.residual_blocks = nn.Sequential(*res_blocks)
        final_blocks = []
        # final block
        for j in range(layers[-1]):
            if j == 0:
                strides = 2
                final_blocks.append(OctconvBottleneck(in_channels=num_filters // 2, out_channels=num_filters,
                                alpha=alpha, stride=strides, downsample_shortcut=True,
                                last_block=True, expansion=expansion))
            else:
                strides = 1
                final_blocks.append(Bottleneck(in_channels=num_filters * expansion, out_channels=num_filters,
                                stride=strides, expansion=expansion))

        self.final_blocks = nn.Sequential(*final_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters * expansion, classes)

        # weights initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)#, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        x = self.first_block(x)
        x = self.residual_blocks(x)
        x = self.final_blocks(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

class OctFiResNet(OctaveResNet):
    def __init__(self, layers=[4, 2],
                 classes=2,
                 alpha=(.25, .25),
                 expansion=4,
                 initial_filters=64,
                 initial_strides=False,
                 **kwargs):
        super(OctFiResNet, self).__init__(layers=layers,
                classes=classes,
                alpha=alpha,
                expansion=expansion,
                initial_filters=initial_filters,
                initial_strides=initial_strides,
                 **kwargs)

# end OctFiResNet
