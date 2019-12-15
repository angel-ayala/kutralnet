import os
import numpy as np
import torch
import argparse
from contextlib import redirect_stdout
from datasets import FireImagesDataset
from utils.training import train_model, plot_history, save_history
from utils.models import models_conf

from models.firenet_pt import FireNet
from models.octfiresnet import OctFiResNet
from models.resnet import resnet_sharma
# from models.kutralnet import KutralNet

parser = argparse.ArgumentParser(description='Fire classification training')
parser.add_argument('--base_model', metavar='BM', default='kutralnet',
                    help='modelo a entrenar')
parser.add_argument('--epochs', metavar='E', default=100, type=int,
                    help='number of maximum iterations')
parser.add_argument('--preload_data', metavar='PD', default=False, type=bool,
                    help='cargar dataset on-memory')
parser.add_argument('--dataset', metavar='D', default='FiSmo',
                    help='seleccion de dataset')
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
base_model = args.base_model#'octfiresnet'
# train config
batch_size = 32
epochs = args.epochs#100
shuffle_dataset = True
preload_data = bool(args.preload_data)#False # load dataset on-memory
# model pre-configuration
config = models_conf[base_model]
img_dims = config['img_dims']
model_name = config['model_name']

# common preprocess
transform_compose = config['preprocess']

# dataset read
dataset_name = args.dataset + 'Dataset'
data_path = os.path.join('.', 'datasets', dataset_name)
train_data = FireImagesDataset(name=args.dataset, root_path=data_path,
            transform=transform_compose, preload=preload_data)
val_data = FireImagesDataset(name=args.dataset, root_path=data_path,
            purpose='test', transform=transform_compose, preload=preload_data)

num_classes = len(train_data.labels)

# model selection
if base_model == 'firenet':
    model = FireNet(classes=num_classes)
elif base_model == 'octfiresnet':
    model = OctFiResNet(classes=num_classes)
elif base_model == 'resnet':
    model = resnet_sharma(classes=num_classes)
# elif base_model == 'kutralnet':
#     model = KutralNet(classes=num_classes)
else:
    raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet]')

# optimizers
criterion = config['criterion'] #nn.CrossEntropyLoss()
opt_args = {'params': model.parameters()}#,'eps': 1e-7}
opt_args.update(config['optimizer_params'])
optimizer = config['optimizer'](**opt_args)
# sched_args = {'optimizer': optimizer, 'T_max': 100, 'eta_min':1e-4}
# sched_args.update(config['scheduler_params'])
# scheduler = conf['scheduler'](**sched_args)
scheduler = None

folder_path = os.path.join('.', 'models', 'saved')
# logs
with open(os.path.join(folder_path, 'log_{}.log'.format(base_model)), 'w') as f:
        with redirect_stdout(f):
            print(model)
            # training metrics
            history, best_model, time_elapsed = train_model(model, criterion, optimizer, train_data, val_data,
                        epochs=epochs, batch_size=batch_size, shuffle_dataset=shuffle_dataset, scheduler=scheduler,
                        use_cuda=use_cuda)

# model save
model_path = os.path.join(folder_path, model_name)
print('Saving model {}'.format(model_path))
torch.save(best_model, model_path)
# metrics save
history_path = os.path.join(folder_path, 'history_{}.csv'.format(base_model))
save_history(history, file_path=history_path)

plot_history(history, base_name=base_model, folder_path=folder_path)
