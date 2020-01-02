import os
import numpy as np
import importlib

import torch
from contextlib import redirect_stdout
from torchvision import transforms
from datasets import available_datasets
from utils.training import test_model
from utils.models import models_conf


parser = argparse.ArgumentParser(description='Fire classification test')
parser.add_argument('--base_model', metavar='BM', default='kutralnet',
                    help='modelo a entrenar')
parser.add_argument('--preload_data', metavar='PRELOAD', default=0, type=bool,
                    help='cargar dataset on-memory')
parser.add_argument('--dataset', metavar='D', default='firenet_test',
                    help='seleccion de dataset')
parser.add_argument('--model_version', metavar='MODELVER', default=None,
                    help='seleccion de modelo')
args = parser.parse_args()

# Seed
seed_val = 666
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# choose model
base_model = args.base_model
version = args.model_version
# test config
batch_size = 32
preload_data = bool(args.preload_data) # load dataset on-memory
# model pre-configuration
config = models_conf[base_model]
img_dims = config['img_dims']
model_name = config['model_name'])

# common preprocess
transform_compose = config['preprocess_test']

# dataset read
dataset_name = args.dataset
test_dataset = available_datasets[dataset_name]
dataset = test_dataset(transform=transform_compose, preload=preload_data)

num_classes = len(dataset.labels)

# folder of saved results
final_folder = dataset_name if version is None else '{}_{}'.format(dataset_name, version)
folder_name = os.path.join(base_model, final_folder)
# folder_name = '{}_{}_{}'.format(base_model, dataset_name, version)
folder_path = os.path.join('.', 'models', 'saved', folder_name)

# model selection
if base_model in models_conf:
    module = importlib.import_module(config['module_name'])
    fire_model = getattr(module, config['class_name'])
    model = fire_model(classes=num_classes)
else:
    raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet (and lite variations)]')

model_path = os.path.join(folder_path, model_name)
print('Loading model {}'.format(model_path))
model.load_state_dict(torch.load(model_path))

with open(os.path.join(folder_path, 'training.log'), 'a+') as f:
    with redirect_stdout(f):
        test_model(model, dataset, batch_size=batch_size, use_cuda=use_cuda)
