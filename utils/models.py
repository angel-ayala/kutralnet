from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from .nadam_optim import Nadam
from torchvision import transforms
from datasets import CustomNormalize

models_conf = {
    'firenet': {
        'img_dims': (64, 64),
        'model_name': 'model_firenet.pth',
        'criterion': CrossEntropyLoss(),
        'optimizer': Adam,
        'optimizer_params': {'eps': 1e-6},
        'preprocess': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ])
    },
    'octfiresnet': {
        'img_dims': (96, 96),
        'model_name': 'model_octfiresnet.pth',
        'criterion': CrossEntropyLoss(),
        'optimizer': Nadam,
        'optimizer_params': {'lr': 1e-4, 'eps': 1e-7},
        'preprocess': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ])
    },
    'resnet': {
        'img_dims': (224, 224),
        'model_name': 'model_resnet.pth',
        'criterion': CrossEntropyLoss(),
        'optimizer': Adam,
        'optimizer_params': {},
        'preprocess': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ])
    },
    'kutralnet': {
        'img_dims': (64, 64),
        'model_name': 'model_kutralnet.pth',
        'criterion': CrossEntropyLoss(),
        'optimizer': Adam,
        'optimizer_params': {},
        'preprocess': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor(),
                       CustomNormalize((-1, 1))
                    ])
    }
}
