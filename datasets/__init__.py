from .fire_images_dataset import CustomNormalize
from .fire_images_dataset import FireImagesDataset
from .fire_images_dataset import FireNetDataset
from .fire_images_dataset import FireNetTestDataset
from .fire_images_dataset import FiSmoDataset
from .fire_images_dataset import FiSmoBalancedDataset
from .fire_images_dataset import FiSmoBlackDataset
from .fire_images_dataset import FiSmoBalancedBlackDataset

__all__ = [ 'FireImagesDataset', 'FireNetDataset', 'FireNetTestDataset',
            'FiSmoDataset', 'FiSmoBalancedDataset', 'FiSmoBlackDataset', 'FiSmoBalancedBlackDataset',
            'CustomNormalize' ]

available_datasets = {
    'firenet': {
        'name': 'FireNet',
        'class': FireNetDataset,
        'num_classes': 2
    },
    'firenet_test': {
        'name': 'FireNet Test',
        'class': FireNetTestDataset,
        'num_classes': 2
    },
    'fismo': {
        'name': 'FiSmo',
        'class': FiSmoDataset,
        'num_classes': 2
    },
    'fismo_balanced': {
        'name': 'FiSmoB',
        'class': FiSmoBalancedDataset,
        'num_classes': 2
    },
    'fismo_black': {
        'name': 'FiSmoA',
        'class': FiSmoBlackDataset,
        'num_classes': 2
    },
    'fismo_balanced_black': {
        'name': 'FiSmoBA',
        'class': FiSmoBalancedBlackDataset,
        'num_classes': 2
    }
}
