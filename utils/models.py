from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision import transforms
from .nadam_optim import Nadam

models_conf = {
    'firenet': {
        'img_dims': (64, 64),
        'model_name': 'FireNet',
        'model_path': 'model_firenet.pth',
        'class_name': 'FireNet',
        'module_name': 'models.firenet_pt',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {'eps': 1e-6},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((64, 64)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }
}

models_conf['octfiresnet'] = {
        'img_dims': (96, 96),
        'model_name': 'OctFiResNet',
        'model_path': 'model_octfiresnet.pth',
        'class_name': 'OctFiResNet',
        'module_name': 'models.octfiresnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': Nadam,
        'optimizer_params': {'lr': 1e-4, 'eps': 1e-7},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((96, 96)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['resnet'] = {
        'img_dims': (224, 224),
        'model_name': 'ResNet50',
        'model_path': 'model_resnet.pth',
        'class_name': 'resnet_sharma',
        'module_name': 'models.resnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((224, 224)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['kutralnet'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet',
        'model_path': 'model_kutralnet.pth',
        'class_name': 'KutralNet',
        'module_name': 'models.kutralnet',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': optim.lr_scheduler.StepLR,
        'scheduler_params': { 'step_size':85 }
    }

models_conf['kutralnetoct'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Octave',
        'model_path': 'model_kutralnetoct.pth',
        'class_name': 'KutralNetOct',
        'module_name': 'models.kutralnetoct',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['kutralnet_mobile'] =  {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Mobile',
        'model_path': 'model_kutralnet_mobile.pth',
        'class_name': 'KutralNetMobile',
        'module_name': 'models.kutralnet_mobile',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

models_conf['kutralnet_mobileoct'] = {
        'img_dims': (84, 84),
        'model_name': 'KutralNet Mobile Octave',
        'model_path': 'model_kutralnet_mobileoct.pth',
        'class_name': 'KutralNetMobileOct',
        'module_name': 'models.kutralnet_mobileoct',
        'criterion': CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {},
        'preprocess_train': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_val': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'preprocess_test': transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ]),
        'scheduler': None,
        'scheduler_params': {}
    }

def get_config(base_model):
    return models_conf[base_model]
