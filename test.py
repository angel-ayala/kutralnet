import os
import numpy as np

import torch
from torchvision import transforms
from datasets import FireImagesDataset, CustomNormalize
from utils.training import test_model
from models.octave_resnet import OctFiResNet

# Seed
seed_val = 666
use_cuda = True
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

img_dims = (96, 96)
model_name = 'model_octfiresnet.pth'

# common preprocess
transform_compose = transforms.Compose([
           transforms.Resize(img_dims), #redimension
           transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # values [-1, 1]
           # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # values ~[-1, 1]
           CustomNormalize((0, 1))
        ])

# dataset read
data_path = os.path.join('.', 'datasets', 'FireNetDataset')
dataset = FireImagesDataset(name='FireNet', root_path=data_path, csv_file='test_dataset.csv',
            transform=transform_compose)

# test config
batch_size = 32
num_classes = len(dataset.labels)

model = OctFiResNet(classes=num_classes)
model.load_state_dict(torch.load('models/saved/' + model_name))

test_model(model, dataset, batch_size=batch_size, use_cuda=use_cuda):
