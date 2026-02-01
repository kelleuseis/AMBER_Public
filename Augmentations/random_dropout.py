import numpy as np

from registry import augmentation_registry
from Augmentations.base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("random_dropout")
class RandomDropout(BaseAugmentation):
    '''
    Random station/channel dropout (replaced with zeroed trace)
    
    Parameters
    ---------------------------------------
    droupout_nsta: int
        number of stations to dropout
        
    all_dropout_chance: float
        probability [0, 1] of applying station droupout 
        instead of channel droupout
    '''
    required_params = ["dropout_nsta"]
    optional_params = {"all_dropout_chance":0.5}
    scope = "windowed"

    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        if np.random.random() < self.augment_chance:
            nsta = waves.shape[0]
            dropout_nsta = min(self.dropout_nsta, nsta)
            dropout_idxs = np.random.choice(nsta, size=dropout_nsta, replace=False)
            
            if np.random.random() < self.all_dropout_chance:
                waves[dropout_idxs, :, :] = 0.0
                pickarr[dropout_idxs, :, :] = np.nan
            else:
                waves[dropout_idxs, 0:2, :] = 0.0

        return waves, pickarr, stnout, []  