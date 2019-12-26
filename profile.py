import torch
from thop import profile, count_hooks, clever_format
from models.firenet_pt import FireNet
from models.octfiresnet import OctFiResNet
from models.resnet import resnet_sharma
from models.kutralnet import KutralNet
from models.kutralnetoct import KutralNetOct
from models.kutralnet_mobile import KutralNetMobile
from models.kutralnet_mobileoct import KutralNetMobileOct
from utils.models import models_conf, get_config

num_classes = 2
# choose model
base_model = 'firenet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = FireNet(classes=num_classes)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('FireNet', img_dims, 'flops, params', flops, params)

# choose model
base_model = 'octfiresnet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = OctFiResNet(classes=num_classes)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('OctFiResNet', img_dims, 'flops, params', flops, params)

# choose model
base_model = 'resnet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = resnet_sharma(classes=num_classes)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('ResnetMod', img_dims, 'flops, params', flops, params)

# choose model
base_model = 'kutralnet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = KutralNet(classes=num_classes)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('KutralNet', img_dims, 'flops, params', flops, params)

# choose model
base_model = 'kutralnet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = KutralNetOct(classes=num_classes)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('KutralNetOct', img_dims, 'flops, params', flops, params)

# choose model
base_model = 'kutralnet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = KutralNetMobile(classes=num_classes)
# print('KutralNetLite', model)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('KutralNetMobile', img_dims, 'flops, params', flops, params)

# choose model
base_model = 'kutralnet'
# model pre-configuration
img_dims = get_config(base_model)['img_dims']

model = KutralNetMobileOct(classes=num_classes)
# print('KutralNetLiteOct', model)
input = torch.randn(1, 3, img_dims[0], img_dims[1])
flops, params = profile(model, verbose=False,
        inputs=(input, ),
        custom_ops={torch.nn.Dropout2d: None})
flops, params = clever_format([flops, params], "%.3f")
print('KutralNetMobileOct', img_dims, 'flops, params', flops, params)
