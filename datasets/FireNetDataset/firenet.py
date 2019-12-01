import os
import math
import glob
import pandas as pd

from datasets.dataset import Dataset

class FireNetDataset(Dataset):
    """
    Dataset available at https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq
    Train filename: Training Dataset.zip
    Test filename: Test_Dataset1__Our_Own_Dataset.zip

    Worked at 'FireNet: A Specialized Lightweight Fire & SmokeDetection Model for Real-Time IoT Applications'
    https://arxiv.org/pdf/1905.11922.pdf
    """
    def __init__(self, size=(224, 224), debug=False):
        super().__init__('FireNet', 'FireNetDataset', size=size, debug=debug)
    # end __init__

    def prepare_training(self):
        print('Preparing training images...')
        # train path
        firenet_dt = os.path.join(self.path, 'Training')
        # FireNet fire
        fire_path = os.path.join(firenet_dt, 'Fire')

        # fire dataframe
        fire_df = pd.DataFrame()
        fire_df['image_id'] = os.listdir(fire_path)
        fire_df['class'] = self.classes['fire']['label']
        fire_df.insert(0, 'folder_path', os.path.join('Training', 'Fire'))
        fire_df.drop_duplicates(subset='image_id', inplace=True)

        # fire dataset validation split
        test_size = .3
        test_qt = math.ceil(len(fire_df) * test_size)
        fire_df = self.split_dataframe(fire_df, test_qt)

        # FireNet no fire
        no_fire_path = os.path.join(firenet_dt, 'NoFire')

        # fire dataframe
        no_fire_df = pd.DataFrame()
        no_fire_df['image_id'] = os.listdir(no_fire_path)
        no_fire_df['class'] = self.classes['no_fire']['label']
        no_fire_df.insert(0, 'folder_path', os.path.join('Training', 'NoFire'))
        no_fire_df.drop_duplicates(subset='image_id', inplace=True)

        # no fire dataset validation split
        test_size = .3
        test_qt = math.ceil(len(no_fire_df) * test_size)
        no_fire_df = self.split_dataframe(no_fire_df, test_qt)

        # concat splitted dataset
        firenet_df = pd.concat([fire_df, no_fire_df])

        # order data reset
        firenet_df.sort_values(['folder_path', 'image_id'], inplace=True)
        firenet_df.reset_index(drop=True, inplace=True)

        # summary
        print(firenet_df.groupby(['class', 'purpose']).agg({'purpose': ['count']}))

        # freeze processing saving at csv
        self.save_dataframe(firenet_df, filename='dataset.csv')
    # end preprocess_train

    def prepare_testing(self):
        print('Preparing testing images...')
        # FireNet test fire
        test_path = os.path.join(self.path, 'Test')
        fire_folders = glob.glob(os.path.join(test_path, 'Fire*'))

        # fire dataframe
        fire_df = pd.DataFrame(columns=['folder_path', 'image_id', 'class'])
        for fire_folder in fire_folders:
            fire_path = fire_folder.split(os.path.sep)[-1]
            # tmp storage
            tmp_df = pd.DataFrame()
            tmp_df['image_id'] = os.listdir(fire_folder)
            tmp_df['class'] = self.classes['fire']['label']
            tmp_df.insert(0, 'folder_path', os.path.join('Test', fire_path))
            fire_df = pd.concat([fire_df, tmp_df])

        # FireNet test no fire
        no_fire_folders = glob.glob(os.path.join(test_path, 'NoFire*'))

        # fire dataframe
        no_fire_df = pd.DataFrame()
        for no_fire_folder in no_fire_folders:
            no_fire_path = no_fire_folder.split(os.path.sep)[-1]
            # tmp storage
            tmp_df = pd.DataFrame()
            tmp_df['image_id'] = os.listdir(no_fire_folder)
            tmp_df['class'] = self.classes['no_fire']['label']
            tmp_df.insert(0, 'folder_path', os.path.join('Test', no_fire_path))
            no_fire_df = pd.concat([no_fire_df, tmp_df])

        # concat splitted dataset
        firenet_test_df = pd.concat([fire_df, no_fire_df])

        # order data reset
        firenet_test_df.sort_values(['folder_path', 'image_id'], inplace=True)
        firenet_test_df.reset_index(drop=True, inplace=True)

        # summary
        print(firenet_test_df['class'].value_counts())

        # freeze processing saving at csv
        self.save_dataframe(firenet_test_df, filename='test_dataset.csv')
    # end preprocess_test
