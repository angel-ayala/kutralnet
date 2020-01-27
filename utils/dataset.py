# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd

def read_and_resize(img_path, resize=(224, 224), color_fix=True):
    img = cv2.imread(img_path)
    if color_fix:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not resize is None and isinstance(resize, tuple):
        img = cv2.resize(img, resize)
    else:
        raise ValueError('resize must be a tuple (width, height)')
    return img
# end read_and_resize

def read_dataset_from_csv(dataset_path, csv_name='dataset.csv', resize=None, val_split=True, debug=False):
    # dataset storage
    x_train, y_train = [], []
    if val_split:
        x_test, y_test = [], []

    # csv read
    dataset_df = pd.read_csv(os.path.join(dataset_path, csv_name))

    print('Loading images...')
    # rows iterate and image load
    for index, row in dataset_df.iterrows():
        path = os.path.join(dataset_path, row['folder_path'], row['image_id'])
        if debug:
            print(path, end=' ')
        img = read_and_resize(path, resize=resize)
        target = 1 if row['class'] == 'fire' else 0

        if val_split and row['purpose'] == 'val':
            x_test.append(img)
            y_test.append(target)
        else:
            x_train.append(img)
            y_train.append(target)

        if debug:
            print('ok')

    print('Ok')
    if val_split:
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    else:
        return np.array(x_train), np.array(y_train)
# read_dataset_from_csv

def load_fismo_dataset(fismo_path, val_split=True, resize=(224, 224), debug=False):
    # load images
    print('Loading FiSmo with', resize, 'dimension')
    return read_dataset_from_csv(fismo_path,
                            val_split=val_split,
                            resize=resize,
                            debug=debug)
# end load_fismo_dataset

def load_fismo_black_dataset(fismo_path, val_split=True, resize=(224, 224), debug=False):
    # load images
    print('Loading FiSmoA with', resize, 'dimension')
    return read_dataset_from_csv(fismo_path,
                            val_split=val_split,
                            resize=resize,
                            csv_name='dataset_black.csv',
                            debug=debug)
# end load_fismo_black_dataset

def load_firenet_dataset(firenet_path, val_split=True, resize=(224, 224), debug=False):
    # load images
    print('Loading FireNet with', resize, 'dimension')
    return read_dataset_from_csv(firenet_path,
                                val_split=val_split,
                                resize=resize,
                                debug=debug)
# end load_firenet_dataset

def load_firenet_test_dataset(firenet_path, val_split=False, resize=(224, 224), debug=False):
    # load images
    print('Loading FireNet-Test with', resize, 'dimension')
    return read_dataset_from_csv(firenet_path,
                                val_split=val_split,
                                resize=resize,
                                csv_name='dataset_test.csv',
                                debug=debug)
# end load_firenet_test_dataset

def load_dataset(dt_id='firenet'):
    if dt_id == 'firenet':
        dt_folder = 'FireNetDataset'
        dt_fn = load_firenet_dataset
    elif 'fismo' in dt_id:
        dt_folder = 'FiSmoDataset'
        if dt_id == 'fismo_black':
            dt_fn = load_fismo_black_dataset
        elif dt_id == 'fismo':
            dt_fn = load_fismo_dataset
        else:
            raise ValueError('The dataset must be fismo or fismo_black')
    else:
        raise ValueError('The dataset must be firenet, fismo or fismo_black')

    return dt_folder, dt_fn

# end load_dataset

def preprocess(batch):
    return batch.astype('float32') / 255.
# end preprocess
