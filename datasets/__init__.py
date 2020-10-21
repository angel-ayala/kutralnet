# from .base import BaseDataset
from .utils import CustomNormalize
from .utils import SimpleResizer
from .utils import FireDetectionPreprocess
from .utils import ResizeImageNetPolicy

from torchvision import transforms
from .autoaugment  import ImageNetPolicy

from .fire import FireNetDataset
from .fire import FireNetTestDataset
from .fire import FiSmoDataset
from .fire import FiSmoBalancedDataset
from .fire import FiSmoBlackDataset
from .fire import FiSmoBalancedBlackDataset

from .fire import FireNetRDataset
from .fire import FireNetTestRDataset
from .fire import FiSmoRDataset
from .fire import FiSmoBalancedRDataset
from .fire import FiSmoBlackRDataset
from .fire import FiSmoBalancedBlackRDataset

# from .smoke import SmokeDataset
# from .smoke import SmokeTestDataset

from .fire_smoke import FireNetV2Dataset
from .fire_smoke import FireNetTestV2Dataset
from .fire_smoke import FiSmoV2Dataset
from .fire_smoke import FiSmoBlackV2Dataset
from .fire_smoke import FiSmoBalancedV2Dataset
from .fire_smoke import FiSmoBalancedBlackV2Dataset
from .fire_smoke import FireFlameDataset
from .fire_smoke import FireFlameTestDataset

from .combined import FireFlameV2Dataset
from .combined import FireFlameTestV2Dataset
from .imagenet import ImageNetDataset


__all__ = [ 'ImagePreprocess', 'CustomNormalize',
            'FireNetDataset', 'FireNetTestDataset',
            'FiSmoDataset', 'FiSmoBlackDataset', 
            'FiSmoBalancedDataset', 'FiSmoBalancedBlackDataset',
            'FireNetRDataset', 'FireNetTestRDataset',
            'FiSmoRDataset', 'FiSmoBlackRDataset', 
            'FiSmoBalancedRDataset', 'FiSmoBalancedBlackRDataset',
            'ImageNetDataset', 
            'FireNetV2Dataset', 'FireNetTestV2Dataset',
            'FiSmoV2Dataset', 'FiSmoBlackV2Dataset',
            'FiSmoBalancedV2Dataset', 'FiSmoBalancedBlackV2Dataset', 
            'FireFlameDataset', 'FireFlameTestDataset',
            # 'SmokeDataset', 'SmokeTestDataset',
            'FireFlameV2Dataset', 'FireFlameTestV2Dataset']

# registered preprocess
preprocessing = dict()
# preprocessing['resize'] = transforms.Resize#SimpleResizer
preprocessing['resize'] = (transforms.Resize, None, 'tensor')
# preprocessing['resize_policy'] = ResizeImageNetPolicy
preprocessing['resize_policy'] = (transforms.Resize, 'auto_imagenet', 'tensor')
preprocessing['fire_detection'] = (transforms.Resize, None, 
                                   ('imagenet', [dict(), 
                                                 dict(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229,0.224, 0.225])
                                                 ]))

augmentation = dict()
augmentation['auto_imagenet'] = ImageNetPolicy

postprocessing = dict()
postprocessing['tensor'] = transforms.ToTensor
postprocessing['imagenet'] = [transforms.ToTensor, transforms.Normalize]

# registered datasets
datasets = dict()
# first set of datasets
datasets['firenet'] = FireNetDataset
datasets['firenet_test'] = FireNetTestDataset
datasets['fismo'] = FiSmoDataset
datasets['fismo_black'] = FiSmoBlackDataset
datasets['fismo_balanced'] = FiSmoBalancedDataset
datasets['fismo_balanced_black'] = FiSmoBalancedBlackDataset
# imagenet dataset
datasets['imagenet']= ImageNetDataset
# relabeled fire (& smoke) datasets
datasets['firenet_relabeled'] = FireNetRDataset
datasets['firenet_test_relabeled'] = FireNetTestRDataset
datasets['fismo_relabeled'] = FiSmoRDataset
datasets['fismo_black_relabeled'] = FiSmoBlackRDataset
datasets['fismo_balanced_relabeled'] = FiSmoBalancedRDataset
datasets['fismo_balanced_black_relabeled'] = FiSmoBalancedBlackRDataset
# smoke datasets
# datasets['smokeset']= SmokeDataset
# datasets['smokeset_test']= SmokeTestDataset
# fire and smoke datasets
datasets['firenet_v2']= FireNetV2Dataset
datasets['firenet_testv2']= FireNetTestV2Dataset
datasets['fismo_v2']= FiSmoV2Dataset
datasets['fismo_blackv2']= FiSmoBlackV2Dataset
datasets['fireflame']= FireFlameDataset
datasets['fireflame_test']= FireFlameTestDataset
# mixed fire and smoke datasets
datasets['fireflame_v2']= FireFlameV2Dataset
datasets['fireflame_testv2']= FireFlameTestV2Dataset


def get_preprocessing(preprocess_id, params=None):
    if preprocess_id in preprocessing:
        preprocess, augmentate, postprocess = preprocessing[preprocess_id]
        
        # preprocessing
        if not params is None:
            prep = preprocess(**params)
            
        # augmentation
        if not augmentate is None:
            augment = augmentation[augmentate]()
        else:
            augment = augmentate
            
        # postprocessing
        if isinstance(postprocess, tuple):
            post = []
            for i in range(len(postprocess[1])):
                post.append(postprocessing[postprocess[0]][i](**postprocess[1][i]))
            post = transforms.Compose(post)
        else:
            post = postprocessing[postprocess]()
        
        return prep, augment, post
    else:
        raise ValueError('Must choose a registered preprocessing', preprocessing.keys())


def get_dataset(dataset_id, params=None):
    if dataset_id in datasets:
        dataset_class = datasets[dataset_id]        
        # process params
        if not params is None:
            dataset_params = params
            # process argument flags
            if 'dataset_flags' in dataset_params.keys():
                if not params['dataset_flags'] is None:                    
                    for f in params['dataset_flags']:
                        set_value = not 'no_' in f # flag is negated?     
                        f = f.replace('no_', '') if not set_value else f                            
                        dataset_params[f] = set_value
                        
                # remove to be parsed in class instance
                del dataset_params['dataset_flags']
        
            return dataset_class(**dataset_params)
        else:
            return dataset_class()
    else:
        raise ValueError('Must choose a registered dataset', datasets.keys())
