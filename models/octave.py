import torch.nn as nn
from octconv import OctConv2d

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
    # end __init__

    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            return self.bnh(hf), self.bnl(lf)
        else:
            return self.bnh(x)
    # end forward
# end _BatchNorm2d

class _ReLU(nn.ReLU):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_ReLU, self).forward(hf)
            lf = super(_ReLU, self).forward(lf)
            return hf, lf
        else:
            return super(_ReLU, self).forward(x)
    # end forward
# end _ReLU

class _ReLU6(nn.ReLU6):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_ReLU6, self).forward(hf)
            lf = super(_ReLU6, self).forward(lf)
            return hf, lf
        else:
            return super(_ReLU6, self).forward(x)
    # end forward
# end _ReLU6

class _LeakyReLU(nn.LeakyReLU):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_LeakyReLU, self).forward(hf)
            lf = super(_LeakyReLU, self).forward(lf)
            return hf, lf
        else:
            return super(_LeakyReLU, self).forward(x)
    # end forward
# end _LeakyReLU

class _SELU(nn.SELU):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_SELU, self).forward(hf)
            lf = super(_SELU, self).forward(lf)
            return hf, lf
        else:
            return super(_SELU, self).forward(x)
    # end forward
# end _SELU

class _AvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_AvgPool2d, self).forward(hf)
            lf = super(_AvgPool2d, self).forward(lf)
            return hf, lf
        else:
            return super(_AvgPool2d, self).forward(x)
    # end forward
# end _AvgPool2d

class _MaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        if type(x) is tuple:
            hf, lf = x
            hf = super(_MaxPool2d, self).forward(hf)
            lf = super(_MaxPool2d, self).forward(lf)
            return hf, lf
        else:
            return super(_MaxPool2d, self).forward(x)
    # end forward
# end _MaxPool2d

def _octconv(in_channels, #ip,
            out_channels, #filters,
            kernel_size=3, #(3, 3),
            stride=1, #(1, 1),
            padding='same',
            alpha=(.5, .5),
            dilation=1,
            groups=False,
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
                 dilation=dilation,
                 groups=groups,
                 bias=bias)
# end octconv

def _octconv_bn(in_channels,
                out_channels,
                kernel=3,
                stride=1,
                alpha=(.5, .5),
                padding='same',
                dilation=1,
                groups=False,
                bias=False,
                activation=True,
                act_layer=_ReLU):
    mods = []
    mods.append(_octconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel,
                stride=stride, alpha=alpha, padding=padding, dilation=dilation,
                groups=groups, bias=bias))
    mods.append(_BatchNorm2d(out_channels, alpha=alpha))

    if activation:
        mods.append(act_layer(inplace=True))

    return nn.Sequential(*mods)
# end _octconv_bn

class OctConvBlock(nn.Module):
    def __init__(self, in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                alpha=(.25, .25),
                padding=0,
                dilation=1,
                groups=False,
                bias=False,
                activation=None,
                batch_norm=None):
        super(OctConvBlock, self).__init__()
        self.conv = OctConv2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, alpha=alpha,
                            dilation=dilation, groups=groups, bias=bias)
        self.bn = batch_norm
        self.act = activation
    # end __init__

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    # end forward
# end OctConvBlock
