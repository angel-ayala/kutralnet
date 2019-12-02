import os
import numpy as np
import torch
from datasets import FireImagesDataset
from utils.training import train_model, plot_history
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
config = models_conf[base_model]

img_dims = config['img_dims']
model_name = config['model_name']

# train config
batch_size = 32
shuffle_dataset = True
epochs = 100

# common preprocess
transform_compose = config['preprocess']

# dataset read
data_path = os.path.join('.', 'datasets', 'FireNetDataset')
train_data = FireImagesDataset(name='FireNet', root_path=data_path,
            transform=transform_compose, preload=True)
val_data = FireImagesDataset(name='FireNet', root_path=data_path,
            purpose='test', transform=transform_compose, preload=True)

num_classes = len(train_data.labels)

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

print(model)

# optimizers
criterion = config['criterion'] #nn.CrossEntropyLoss()
opt_args = {'params': model.parameters()}
opt_args.update(config['optimizer_params'])
optimizer = config['optimizer'](**opt_args)

history, best_model = train_model(model, criterion, optimizer, train_data, val_data, epochs=epochs,
            batch_size=batch_size, shuffle_dataset=shuffle_dataset,
            use_cuda=use_cuda)

torch.save(best_model, 'models/saved/' + model_name)
plot_history(history)
