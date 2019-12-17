import os
import numpy as np

import torch
from contextlib import redirect_stdout
from torchvision import transforms
from datasets import FireImagesDataset, CustomNormalize
from utils.training import test_model
from utils.models import models_conf

from models.firenet_pt import FireNet
from models.octfiresnet import OctFiResNet
from models.resnet import resnet_sharma
from models.kutralnet import KutralNet

# Seed
seed_val = 666
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# choose model
base_model = 'kutralnet'
# test config
batch_size = 32
preload_data = True # load dataset on-memory
# model pre-configuration
config = models_conf[base_model]
img_dims = config['img_dims']
model_name = config['model_name']

# common preprocess
transform_compose = config['preprocess']

# dataset read
data_path = os.path.join('.', 'datasets', 'FireNetDataset')
dataset = FireImagesDataset(name='FireNet', root_path=data_path, csv_file='test_dataset.csv',
            transform=transform_compose, preload=preload_data)

num_classes = len(dataset.labels)

# model selection
if base_model == 'firenet':
    model = FireNet(classes=num_classes)
elif base_model == 'octfiresnet':
    model = OctFiResNet(classes=num_classes)
elif base_model == 'resnet':
    model = resnet_sharma(classes=num_classes)
elif base_model == 'kutralnet':
    model = KutralNet(classes=num_classes)
else:
    raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet]')

model_path = os.path.join('.', 'models', 'saved', model_name)
print('Loading model {}'.format(model_path))
model.load_state_dict(torch.load(model_path))

with open(os.path.join(folder_path, 'training.log'), 'a+') as f:
    with redirect_stdout(f):
        test_model(model, dataset, batch_size=batch_size, use_cuda=use_cuda)
