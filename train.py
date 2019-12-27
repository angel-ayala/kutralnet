import os
import time
import torch
import argparse
import importlib
import numpy as np
from contextlib import redirect_stdout
from datasets import available_datasets
from utils.training import train_model
from utils.training import plot_history
from utils.training import save_history
from utils.models import models_conf

parser = argparse.ArgumentParser(description='Fire classification training')
parser.add_argument('--base_model', metavar='BM', default='kutralnet',
                    help='modelo a entrenar')
parser.add_argument('--epochs', metavar='E', default=100, type=int,
                    help='number of maximum iterations')
parser.add_argument('--preload_data', metavar='PD', default=0, type=bool,
                    help='cargar dataset on-memory')
parser.add_argument('--dataset', metavar='D', default='fismo',
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
version = 0
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
transform_train = config['preprocess_train']
transform_val = config['preprocess_val']

# dataset read
dataset_name = args.dataset
base_dataset = available_datasets[dataset_name]
train_data = base_dataset(transform=transform_train, preload=preload_data)
val_data = base_dataset(purpose='val', transform=transform_val, preload=preload_data)

num_classes = len(train_data.labels)

# model selection
if base_model in models_conf:
    module = importlib.import_module(config['module_name'])
    fire_model = getattr(module, config['class_name'])
else:
    raise ValueError('Must choose a model first [firenet, octfiresnet, resnet, kutralnet (and lite variations)]')

# optimizers
criterion = config['criterion'] #nn.CrossEntropyLoss()
opt_args = {'params': model.parameters()}#,'eps': 1e-7}
opt_args.update(config['optimizer_params'])
optimizer = config['optimizer'](**opt_args)
scheduler = None

if conf['scheduler'] is not None:
    sched_args = {'optimizer': optimizer}
    sched_args.update(config['scheduler_params'])
    scheduler = conf['scheduler'](**sched_args)

# folder for save results
folder_name = '{}_{}_{}'.format(base_model, dataset_name, version)
folder_path = os.path.join('.', 'models', 'saved', folder_name)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

print('Initiating training, models will be saved at {}'.format(folder_path))
# logs
with open(os.path.join(folder_path, 'training.log'), 'w') as f:
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
history_path = os.path.join(folder_path, 'history.csv')
save_history(history, file_path=history_path)

plot_history(history, folder_path=folder_path)
