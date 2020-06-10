import torch
from thop import profile, clever_format
from .training import get_model

def profile_model(model, config):
    img_dims = config['img_dims']
    x = torch.randn(1, 3, img_dims[0], img_dims[1])
    flops, params = profile(model, verbose=False,
            inputs=(x, ),
            custom_ops={torch.nn.Dropout2d: None})
    flops, params = clever_format([flops, params], "%.4f")
    print('{}_{}: {} flops, {} params'.format(config['model_name'],
                                             img_dims, flops, params))
    return flops, params

def load_profile_model(base_model, num_classes=2, extra_params=None):
    # model selection
    model, config = get_model(base_model, extra_params=extra_params)
    return profile_model(model, config)

if __name__ == '__main__':
    # FireNet
    # profile_model('firenet')
    # OctFiResNet
    load_profile_model('octfiresnet')
    # ResNet50
    load_profile_model('resnet')
    # ResNet18
    load_profile_model('resnet18')
    # KutralNet
    load_profile_model('kutralnet')
    # KutralNet Mobile
    load_profile_model('kutralnet_mobile')
    # KutralNetOctave no-groups
    load_profile_model('kutralnetoct', extra_params={'groups':False})
    # KutralNetOctave
    load_profile_model('kutralnetoct')
    # KutralNet MobileOctave
    load_profile_model('kutralnet_mobileoct')
