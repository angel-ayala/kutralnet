import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from datasets import available_datasets
from utils.training import test_model
from utils.training import get_model
from utils.training import get_paths
from utils.training import add_bool_arg


parser = argparse.ArgumentParser(description='Fire classification test')
parser.add_argument('--base-model', metavar='BM', default='kutralnet',
                    help='the trained model ID to test')
parser.add_argument('--dataset', metavar='DS', default='fismo',
                    help='the dataset ID used for training')
parser.add_argument('--version', metavar='VER', default=None,
                    help='the training version to perform the test')
parser.add_argument('--batch-size', metavar='BS', default=32, type=int,
                    help='the number of items in the batch')
add_bool_arg(parser, 'preload-data', default=True, help='choose if load or not the dataset on-memory')
args = parser.parse_args()

# Seed
seed_val = 666
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed_val)
np.random.seed(seed_val)
torch_device = 'cpu'

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_device = 'cuda'

# user's selection
base_model = args.base_model
train_dataset_name = args.dataset
version = args.version
preload_data = bool(args.preload_data) # load dataset on-memory
batch_size = args.batch_size

# dataset selection
dataset_name = 'firenet_test'
test_dataset = available_datasets[dataset_name]['class']
num_classes = available_datasets[dataset_name]['num_classes']

# model load
model, config = get_model(base_model, num_classes=num_classes)

# dataset load
transform_compose = config['preprocess_test']
dataset = test_dataset(transform=transform_compose, preload=preload_data)

# read models direclty from the repository's folder
root_path = os.path.join('.')
models_root, models_save_path, models_results_path = get_paths(root_path)
# folder of saved results
final_folder = train_dataset_name if version is None else '{}_{}'.format(
    train_dataset_name, version)
folder_name = os.path.join(base_model, final_folder)
save_path = os.path.join(models_save_path, folder_name)
model_path = os.path.join(save_path, config['model_path'])

with open(os.path.join(save_path, 'testing.log'), 'a+') as f:
    with redirect_stdout(f):        
        print('Loading model from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path, 
                                    map_location=torch.device(torch_device)))
        # test summary
        y_true, y_pred, test_accuracy = test_model(model, dataset, 
                                                   batch_size=batch_size, 
                                                   use_cuda=use_cuda)
        y_score = [y[1] for y in y_pred]
        # Compute ROC curve and ROC area:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_summary = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
        print('Area under the ROC curve', roc_auc)
    
        # test report
        target_names = [ dataset.labels[label]['name'] for label in dataset.labels ]
        print('target_names', target_names)
    
        y_pred_class = np.argmax(y_pred, axis=1)
        class_report = classification_report(y_true, y_pred_class,
                                target_names=target_names)#, output_dict=True)
    
        # print('Confusion Matrix', confusion)
        print('Classification Report')
        print(class_report)
        test_results = classification_report(y_true, y_pred_class,
                                target_names=target_names, output_dict=True)
    
        # print('test_results', test_results)
        summary = pd.DataFrame.from_dict(test_results)
        summary = summary.drop(['support'])
        summary.reset_index(inplace=True)
        summary.rename(columns={'index':'metric'}, inplace=True)
        
        # save results
        summary.to_csv(os.path.join(save_path, 'testing_summary.csv'), 
                       index=False)
        with open(os.path.join(save_path, 'roc_summary.pkl'), 'wb') as f:
            pickle.dump(roc_summary, f, pickle.HIGHEST_PROTOCOL)
