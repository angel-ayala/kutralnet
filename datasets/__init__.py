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
    'firenet': FireNetDataset,
    'firenet_test': FireNetTestDataset,
    'fismo': FiSmoDataset,
    'fismo_balanced': FiSmoBalancedDataset,
    'fismo_black': FiSmoBlackDataset,
    'fismo_balanced_black': FiSmoBalancedBlackDataset
}
