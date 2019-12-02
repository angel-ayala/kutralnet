import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datasets import FireImagesDataset, CustomNormalize, ZeroCentered
from utils.nadam_optim import Nadam
from utils.training import train_model, plot_history
from models.octfiresnet import OctFiResNet

# Seed
seed_val = 666
use_cuda = True
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# img_dims = (68, 68)
img_dims = (96, 96)
model_name = 'model_octfiresnet.pth'

# train config
batch_size = 32
validation_split = .3
shuffle_dataset = True
epochs = 100

# common preprocess
transform_compose = transforms.Compose([
           transforms.Resize(img_dims), #redimension
           transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # values [-1, 1]
           # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # values ~[-1, 1]
           # CustomNormalize((0, 1))
           ZeroCentered()
        ])

# dataset read
data_path = os.path.join('.', 'datasets', 'FireNetDataset')
dataset = FireImagesDataset(name='FireNet', root_path=data_path,
            transform=transform_compose, preload=True)

num_classes = len(dataset.labels)

# model parameters
model = OctFiResNet(classes=num_classes)
print(model)

# optimizers
criterion = nn.CrossEntropyLoss()
optimizer = Nadam(model.parameters())#, lr=0.0001)#, eps=1e-7)#, eps=None)

history, best_model = train_model(model, criterion, optimizer, dataset, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, shuffle_dataset=shuffle_dataset, use_cuda=use_cuda)

torch.save(best_model, 'models/saved/' + model_name)
plot_history(history)
