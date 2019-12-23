import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FireImagesDataset(Dataset):
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False):
        self.root_path = root_path
        self.csv_file = csv_file
        self.name = name
        self.purpose = purpose
        self.transform = transform
        self.preload = preload
        self.data = self.read_csv()

        self.labels = {
            'no_fire': {
                'idx': 0,
                'label': 'no_fire',
                'name': 'NoFire'
            },
            'fire': {
                'idx': 1,
                'label': 'fire',
                'name': 'Fire'
            }
        }

        if self.preload:
            self._preload()
    # end __init__

    def __len__(self):
        return len(self.data)
    # end __len__

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            return self.x[idx], self.y[idx]

        return self._item(idx)
    # end __getitem__

    def _item(self, idx):
        img_path = os.path.join(self.root_path,
                                self.data.iloc[idx]['folder_path'],
                                self.data.iloc[idx]['image_id'])
        image = Image.open(img_path).convert('RGB')
        # image = image
        label = self.data.iloc[idx]['class']
        label = self.labels[label]['idx']
        label = torch.from_numpy(np.array(label))

        if self.transform:
            image = self.transform(image)

        return image, label
    # end _item

    def read_csv(self):
        # csv read
        csv_path = os.path.join(self.root_path, self.csv_file)
        print('Reading from file: {} (on-memory={})'.format(csv_path, self.preload))
        dataset_df = pd.read_csv(csv_path)

        if self.purpose is not None and 'purpose' in dataset_df:
            dataset_df = dataset_df[dataset_df['purpose'] == self.purpose]

        return dataset_df
    # read_csv

    def _preload(self):
        self.x = []
        self.y = []

        for i in range(len(self.data)):
            item = self._item(i)
            self.x.append(item[0])
            self.y.append(item[1])
    # end _preload

# end FireImagesDataset

root_path = os.path.dirname(os.path.abspath(__file__))
class FireNetDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FireNet', root_path=os.path.join(root_path, 'FireNetDataset'),
            csv_file='dataset.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FireNetDataset

class FireNetTestDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FireNet-Test', root_path=os.path.join(root_path, 'FireNetDataset'),
            csv_file='dataset_test.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FireNetTestDataset

class FiSmoDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FiSmo', root_path=os.path.join(root_path, 'FiSmoDataset'),
            csv_file='dataset.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoDataset

class FiSmoBalancedDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FiSmoBalanced', root_path=os.path.join(root_path, 'FiSmoBalancedDataset'),
            csv_file='dataset_balanced.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoBalancedDataset

class FiSmoBlackDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FiSmoBlack', root_path=os.path.join(root_path, 'FiSmoBlackDataset'),
            csv_file='dataset_black.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoBlackDataset

class FiSmoBalancedBlackDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FiSmoBalancedBlack', root_path=os.path.join(root_path, 'FiSmoBalancedBlackDataset'),
            csv_file='dataset_balanced_black.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoBalancedBlackDataset

class CustomNormalize:
    def __init__(self, interval=(0, 1)):
        self.a = interval[0]
        self.b = interval[1]
    # end __init__

    def __call__(self, tensor):
        minimo = tensor.min()
        maximo = tensor.max()
        return (self.b - self.a) * ((tensor - minimo) / (maximo - minimo)) + self.a
    # end __call__

    def __repr__(self):
        return self.__class__.__name__ + '([{}, {}])'.format(self.a, self.b)
    # end __repr__
# end CustomNormalize

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FireNetDataset')

    print('data_path', data_path)

    transform_compose = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    dataset = FireImagesDataset('FireNet', data_path, transform=transform_compose)
    print(dataset.data)
    print(len(dataset))
    sample = dataset[1618]
    print(sample)
