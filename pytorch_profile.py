import torch
from thop import profile, count_hooks, clever_format
from models.firenet_pt import FireNet
from models.lighternet_v3 import LighterNet
from models.octave_resnet import OctFiResNet

model = FireNet(2)
input = torch.randn(1, 3, 64, 64)
flops, params = profile(model,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('FireNet flops, params', flops, params)
print()

model = LighterNet(2)
input = torch.randn(1, 3, 64, 64)
flops, params = profile(model,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None,
        torch.nn.BatchNorm2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('LighterNet flops, params', flops, params)
print()

model = OctFiResNet(classes=2)
input = torch.randn(1, 3, 64, 64)
flops, params = profile(model,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('OctFiResNet flops, params', flops, params)
