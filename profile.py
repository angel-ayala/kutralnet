import torch
from thop import profile, count_hooks, clever_format
from models.octave_resnet import OctFiResNet

model = OctFiResNet(classes=2)
input = torch.randn(1, 3, 64, 64)
flops, params = profile(model,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('OctFiResNet flops, params', flops, params)
