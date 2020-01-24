from .fire_images_dataset import CustomNormalize
from .fire_images_dataset import FireImagesDataset
from .fire_images_dataset import FireNetDataset
from .fire_images_dataset import FireNetTestDataset
from .fire_images_dataset import FiSmoDataset
from .fire_images_dataset import FiSmoBalancedDataset
from .fire_images_dataset import FiSmoBlackDataset
from .fire_images_dataset import FiSmoBalancedBlackDataset

__all__ = ['FireImagesDataset', 'CustomNormalize', 'FireNetDataset', 'FireNetTestDataset', 'FiSmoDataset', 'FiSmoBalancedDataset', 'FiSmoBlackDataset', 'FiSmoBalancedBlackDataset']

available_datasets = {
    'firenet': {
        'name': 'FireNet',
        'class': FireNetDataset
    },
    'firenet_test': {
        'name': 'FireNet Test',
        'class': FireNetTestDataset
    },
    'fismo': {
        'name': 'FiSmo',
        'class': FiSmoDataset
    },
    'fismo_balanced': {
        'name': 'FiSmoB',
        'class': FiSmoBalancedDataset
    },
    'fismo_black': {
        'name': 'FiSmoA',
        'class': FiSmoBlackDataset
    },
    'fismo_balanced_black': {
        'name': 'FiSmoBA',
        'class': FiSmoBalancedBlackDataset
    }
}
