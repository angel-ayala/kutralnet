import torch
import importlib
from thop import profile, count_hooks, clever_format
from utils.models import models_conf, get_config

def profile_model(base_model, num_classes=2, extra_params=None):
    # model selection
    if base_model in models_conf:
        config = get_config(base_model)
        module = importlib.import_module(config['module_name'])
        fire_model = getattr(module, config['class_name'])
        params = {'classes': num_classes }

        if extra_params is not None:
            params.update(extra_params)

        model = fire_model(**params)
    else:
        raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet (and lite variations)]')

    img_dims = config['img_dims']
    input = torch.randn(1, 3, img_dims[0], img_dims[1])
    flops, params = profile(model, verbose=False,
            inputs=(input, ),
            custom_ops={torch.nn.Dropout2d: None})
    flops, params = clever_format([flops, params], "%.3f")
    print(config['class_name'], img_dims, 'flops, params', flops, params)

# FireNet
profile_model('firenet')
# OctFiResNet
profile_model('octfiresnet')
# ResNet
profile_model('resnet')
# KutralNet
profile_model('kutralnet')
# KutralNet Mobile
profile_model('kutralnet_mobile')
# KutralNetOctave no-groups
profile_model('kutralnetoct', extra_params={'groups':False})
# KutralNetOctave
profile_model('kutralnetoct')
# KutralNet MobileOctave
profile_model('kutralnet_mobileoct')
