import os
import numpy as np

import torch
from torchvision import transforms
from datasets import FireImagesDataset, CustomNormalize
from utils.training import test_model
from utils.models import models_conf
from models.firenet_pt import FireNet

# Seed
seed_val = 666
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# choose model
base_model = 'firenet'
config = models_conf[base_model]

img_dims = config['img_dims']
model_name = config['model_name']

# common preprocess
transform_compose = config['preprocess']

# dataset read
data_path = os.path.join('.', 'datasets', 'FireNetDataset')
dataset = FireImagesDataset(name='FireNet', root_path=data_path, csv_file='test_dataset.csv',
            transform=transform_compose, preload=True)

# test config
batch_size = 32
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

model.load_state_dict(torch.load('models/saved/' + model_name))

test_model(model, dataset, batch_size=batch_size, use_cuda=use_cuda)
