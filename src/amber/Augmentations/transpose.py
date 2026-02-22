import numpy as np

from amber.registry import augmentation_registry
from .base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("transpose")
class Transpose(BaseAugmentation):
    '''
    Transpose waves from (nsta, nchnl, ndp) to
    (nchnl, nsta, ndp).
    '''
    scope = "windowed"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        waves = waves.transpose(1, 0, 2)
        return waves, pickarr, stnout, []