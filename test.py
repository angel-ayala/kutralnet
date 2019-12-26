import os
import numpy as np

import torch
from contextlib import redirect_stdout
from torchvision import transforms
from datasets import available_datasets
from utils.training import test_model
from utils.models import models_conf

from models.firenet_pt import FireNet
from models.octfiresnet import OctFiResNet
from models.resnet import resnet_sharma
from models.kutralnet import KutralNet
from models.kutralnetoct import KutralNetOct
from models.kutralnet_mobile import KutralNetMobile
from models.kutralnet_mobileoct import KutralNetMobileOct

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
version = 0
# test config
batch_size = 32
preload_data = True # load dataset on-memory
# model pre-configuration
config = models_conf[base_model]
img_dims = config['img_dims']
model_name = config['model_name'])

# common preprocess
transform_compose = config['preprocess_test']

# dataset read
dataset_name = 'firenet_test'
test_dataset = available_datasets[dataset_name]
dataset = test_dataset(transform=transform_compose, preload=preload_data)

num_classes = len(dataset.labels)

# folder of save results
folder_name = '{}_{}_{}'.format(base_model, dataset_name, version)
folder_path = os.path.join('.', 'models', 'saved', folder_name)

# model selection
if base_model == 'firenet':
    model = FireNet(classes=num_classes)
elif base_model == 'octfiresnet':
    model = OctFiResNet(classes=num_classes)
elif base_model == 'resnet':
    model = resnet_sharma(classes=num_classes)
elif base_model == 'kutralnet':
    model = KutralNet(classes=num_classes)
elif base_model == 'kutralnetoct':
    model = KutralNetOct(classes=num_classes)
elif base_model == 'kutralnet_mobile':
    model = KutralNetMobile(classes=num_classes)
elif base_model == 'kutralnet_mobileoct':
    model = KutralNetMobileOct(classes=num_classes)
else:
    raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet]')

model_path = os.path.join(folder_path, model_name)
print('Loading model {}'.format(model_path))
model.load_state_dict(torch.load(model_path))

with open(os.path.join(folder_path, 'training.log'), 'a+') as f:
    with redirect_stdout(f):
        test_model(model, dataset, batch_size=batch_size, use_cuda=use_cuda)
