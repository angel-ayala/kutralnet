import os
import torch
import argparse
import numpy as np
from contextlib import redirect_stdout
from datasets import available_datasets
from utils.training import SaveCallback
from utils.training import train_model
from utils.training import plot_history
from utils.training import save_history
from utils.training import get_model
from utils.training import get_paths
from utils.training import add_bool_arg


parser = argparse.ArgumentParser(description='Classification models training script')
parser.add_argument('--base-model', metavar='BM', default='kutralnet',
                    help='the model ID for training')
parser.add_argument('--epochs', metavar='EP', default=100, type=int,
                    help='the number of maximum iterations')
parser.add_argument('--batch-size', metavar='BS', default=32, type=int,
                    help='the number of items in the batch')
parser.add_argument('--dataset', metavar='DS', default='fismo',
                    help='the dataset ID for training')
parser.add_argument('--version', metavar='VER', default=None,
                    help='the training version')
add_bool_arg(parser, 'preload-data', default=False, help='choose if load or not the dataset on-memory')
add_bool_arg(parser, 'pin-memory', default=False, help='choose if pin or not the data into CUDA memory')
args = parser.parse_args()


# Seed
seed_val = 666
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# user's selections
base_model = args.base_model
dataset_name = args.dataset
version = args.version
# train config
epochs = args.epochs#100
batch_size = args.batch_size
shuffle_dataset = True
preload_data = bool(args.preload_data)#False # load dataset on-memory
pin_memory = bool(args.pin_memory)#False # pin dataset on-memory

# dataset selection
base_dataset = available_datasets[dataset_name]['class']
num_classes = available_datasets[dataset_name]['num_classes']

# model load
model, config = get_model(base_model, num_classes=num_classes)
# model pre-configuration
img_dims = config['img_dims']
model_path = config['model_path']

# common preprocess
transform_train = config['preprocess_train']
transform_val = config['preprocess_val']

# dataset read
train_data = base_dataset(transform=transform_train, preload=preload_data)
val_data = base_dataset(purpose='val', transform=transform_val, preload=preload_data)

# optimizers
criterion = config['criterion'] #nn.CrossEntropyLoss()
opt_args = {'params': model.parameters()}#,'eps': 1e-7}
opt_args.update(config['optimizer_params'])
optimizer = config['optimizer'](**opt_args)
scheduler = None

if config['scheduler'] is not None:
    sched_args = {'optimizer': optimizer}
    sched_args.update(config['scheduler_params'])
    scheduler = config['scheduler'](**sched_args)

# folder for save results
# save models direclty in the repository's folder
root_path = os.path.join('.')
models_root, models_save_path, models_results_path = get_paths(root_path)

final_folder = dataset_name if version is None else '{}_{}'.format(dataset_name, version)
folder_name = os.path.join(base_model, final_folder)
save_path = os.path.join(models_save_path, folder_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('Initiating training, models will be saved at {}'.format(save_path))
save_callback = SaveCallback(save_path)
# logs
with open(os.path.join(save_path, 'training.log'), 'w') as f:
        with redirect_stdout(f):
            print(model)
            # training metrics
            history, best_model, time_elapsed = train_model(model, criterion, optimizer, train_data, val_data,
                        epochs=epochs, batch_size=batch_size, shuffle_dataset=shuffle_dataset, scheduler=scheduler,
                        use_cuda=use_cuda, pin_memory=pin_memory, callbacks=[save_callback])

# model save
model_path = os.path.join(save_path, model_path)
print('Saving model {}'.format(model_path))
torch.save(best_model, model_path)
# metrics save
save_history(history, file_path=os.path.join(save_path, 'history.csv'))

plot_history(history, folder_path=save_path)
