import numpy as np

from registry import augmentation_registry
from Augmentations.base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("random_flip")
class RandomFlip(BaseAugmentation):
    '''
    Randomly flip the order of stations
    '''
    scope = "windowed"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        if np.random.random() < self.augment_chance:
            waves = np.flip(waves, axis=0).copy()
            pickarr = np.flip(pickarr, axis=0).copy()
            stnout = np.flip(stnout, axis=0).copy()
            
        return waves, pickarr, stnout, []