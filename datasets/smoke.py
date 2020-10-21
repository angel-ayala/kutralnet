#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:49:51 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import os
from torchvision import transforms

from datasets.base import BaseDataset
from datasets.base import datasets_path

class SmokeImages(BaseDataset):
    def __init__(self, 
                 name, 
                 root_path, 
                 csv_file='dataset.csv', 
                 transform=None,
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(SmokeImages, self).__init__(
                name=name, 
                root_path=root_path, 
                csv_file=csv_file, 
                transform=transform,
                purpose=purpose, 
                preload=preload, 
                one_hot=one_hot
            )
    # end __init__
    
    def set_labels(self):
        self.labels = {
            'none': {
                'idx': 0,
                'label': 'none',
                'name': 'None'
            },
            'smoke': {
                'idx': 2,
                'label': 'smoke',
                'name': 'Smoke'
            }
        }
        
    # end set_labels
# end FireSmokeImagesDataset


class SmokeDataset(SmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(SmokeDataset, self).__init__(
            name='Smoke', 
            root_path=os.path.join(datasets_path, 'SmokeDataset'),
            csv_file='dataset.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end SmokeDataset

class SmokeTestDataset(SmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='test', 
                 preload=False, 
                 one_hot=False):
        super(SmokeTestDataset, self).__init__(
            name='Smoke Test', 
            root_path=os.path.join(datasets_path, 'SmokeDataset'),
            csv_file='dataset_test.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end SmokeTestDataset

if __name__ == '__main__':
    data_path = os.path.join(datasets_path, 'SmokeDataset')

    print('data_path', data_path)

    transform_compose = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    dataset = SmokeImages('Smoke', data_path, transform=transform_compose, purpose=None)
    sample = dataset[48]
    print('sample', sample[0].shape, sample[1])
    dataset.labels_describe()
    
    # datasets summaries
    dataset = SmokeDataset(purpose=None) # get all images
    print(dataset)
    if not dataset.is_splitted:
        dataset.split(persist=True)
    dataset.labels_describe()
    
    dataset = SmokeTestDataset(purpose=None) # get all images
    print(dataset)
    dataset.labels_describe()
    