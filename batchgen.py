#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 02:32:46 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import os
import argparse
from datetime import datetime
from models import models
from models import get_model_paths
from datasets import datasets
from utils.training import add_bool_arg


parser = argparse.ArgumentParser(description='Batch bash script generator for training and testing the models.')
parser.add_argument('script', 
                    help='the python script name to be called, can be \'train\' or \'test\'',
                    choices=['train', 'test'])
parser.add_argument('--models', default=['kutralnet'], nargs='+',
                    help='the trained models ID presented, i.e, --models \
                    resnet kutralnet kutralnet_octave ocfiresnet')
parser.add_argument('--model-params', default=None,
                    help='the params to instantiate the model')
parser.add_argument('--activation', default='ce_softmax',
                    help='the activation function for the model')
parser.add_argument('--loss-params', default=None, nargs='*',
                    help='the params to instantiate the cost function as KEY=VAL')
parser.add_argument('--datasets', default=['fismo'], nargs='+',
                    help='the datasets ID used for training as --datasets \
                        fismo (firenet (fismo_black))')
parser.add_argument('--dataset-test', metavar='DATASET_TEST_ID', default='firenet_test',
                    help='the dataset ID for test')  
parser.add_argument('--dataset-flags', default=None, nargs='*',
                    help='the datasets flags to instaciate the dataset, this \
                        flags can be: \
                            - (no_)one_hot: to one-hot encode or not the labels.\
                            - (no_)distributed: to use or not a distributed representation.\
                            - (no_)multi_label: to allow or not the use of multi-label images.')
parser.add_argument('--epochs', metavar='EP', default=100, type=int,
                    help='the number of maximum iterations')
parser.add_argument('--batch-size', metavar='BS', default=32, type=int,
                    help='the number of items in the batch')
parser.add_argument('--versions', nargs='*',
                    help='the training version')
parser.add_argument('--models-path', default='models',
                    help='the path where are stored the models')
parser.add_argument('--output', default='.',
                    help='the output file folder')
add_bool_arg(parser, 'preload-data', default=True, help='choose if load or not the dataset on-memory')
add_bool_arg(parser, 'pin-memory', default=False, help='choose if pin or not the data into CUDA memory')
add_bool_arg(parser, 'seed', default=False, help='choose if set or not a seed for random values')
args = parser.parse_args()


# user's selections
purpose = args.script #'train'
models_id = args.models #'kutralnet'
datasets_id = args.datasets #'fismo'
dataset_test_id = args.dataset_test #'firenet_test'
versions = args.versions #None    
output_folder = args.output #'.'
# process config
activation_fn = args.activation # 'softmax'
epochs = args.epochs #100
batch_size = args.batch_size #32
preload_data = bool(args.preload_data) #False # load dataset on-memory
pin_memory = bool(args.pin_memory) #False # pin dataset on-memory
must_seed = bool(args.seed) #True # set seed value
models_root = args.models_path
model_params = args.model_params
dataset_flags = args.dataset_flags
loss_params = args.loss_params

if " " in models_root:
    models_root_str = models_root.replace(" ", "\ ")
else:
    models_root_str = models_root

if versions is None or len(versions) == 0:
     versions = [None]

# command line options
common_options = "--batch-size {} ".format(batch_size)

if purpose == 'train':
    common_options += "--epochs {} ".format(epochs)
    common_options += "--pin-memory " if pin_memory else "--no-pin-memory "

# model params
model_options = "--models-path {} ".format(models_root_str)

if not model_params is None:
    model_options += "--model-params {} ".format(model_params)
    
# dataset
dataset_options = ""
    
if purpose == 'test':
    dataset_options += "--dataset-test {} ".format(dataset_test_id)   
   
if not dataset_flags is None:
    flags = ''
    for f in dataset_flags:
        flags += f + ' '
        
    dataset_options += "--dataset-flags {} ".format(flags)
    
# activation and loss function
activation_options = "--activation {} ".format(activation_fn)

if not loss_params is None and purpose == 'train':
    activation_options += "--loss-params {} ".format(loss_params)
    
common_options += "--preload-data " if preload_data else "--no-preload-data "
common_options += "--seed " if must_seed else "--no-seed "

# bash script
bash_name = "{}_script.sh".format(purpose)
bash_file = os.path.join(output_folder, bash_name)
print('Writing in', os.path.abspath(bash_file))

bash_file = open(bash_file, "w")
bash_file.write("#!/bin/bash\n") # init line
comment = "# This file was autogenerated on {}\n".format(
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
bash_file.write(comment)

print('Models path:', os.path.abspath(models_root))

for model_id in models_id:
    for dataset_id in datasets_id:
        for ver in versions:
            # check if exists
            if not model_id in models:
                print(model_id, 'model not registered!, passing...')
                continue
            
            if not dataset_id in datasets:
                print(dataset_id, 'dataset not registered!, passing...')
                continue
            
            # check if trained on test purpose
            if purpose == "test":
                save_path, _ = get_model_paths(models_root, model_id, dataset_id,
                                               version=ver)
                if save_path is None:
                    print('Missing training data for', 
                          model_id, dataset_id, ver)
                    continue
                
            command = "python {}.py ".format(purpose)
            command += "--model {} ".format(model_id)
            command += model_options
            
            command += "--dataset {} ".format(dataset_id)
            command += dataset_options
            
            command += activation_options            
            
            if not ver is None:
                command += "--version {} ".format(ver)
                
            command += common_options
            
            bash_file.write("echo Executing: " + command + "\n")
            bash_file.write(command + "\n")

bash_file.close()
os.chmod(bash_name, 0o0755)