import numpy as np

from amber.registry import augmentation_registry
from .base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("random_rotation")
class RandomRotation(BaseAugmentation):
    '''
    Randomly apply horizontal or 3D rotation
    
    Parameters
    -------------------------------------
    mode: '2d' or '3d'
        mode of rotation
    '''
    required_params = ["mode"]
    scope = "windowed"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        waves_aug = waves.copy()
        nsta, _, ndp = waves.shape
        
        if np.random.random() < self.augment_chance:
            rotation = np.radians(np.random.uniform(0, 360))
            
            if self.mode == "2d":
                RH = np.array([[np.cos(rotation), -np.sin(rotation)],
                               [np.sin(rotation), np.cos(rotation)]])
                flattenedH = waves[:, 0:2, :].transpose(1, 0, 2).reshape(2, -1)
                rotatedH = (RH@flattenedH).reshape(2, nsta, ndp).transpose(1, 0, 2)
                
                waves_aug[:, 0:2, :] = rotatedH
                
            elif self.mode == "3d":
                chnl = np.random.normal(size=3)
                chnl /= np.linalg.norm(chnl)

                K = np.array([[0, -chnl[2], chnl[1]],
                              [chnl[2], 0, -chnl[0]],
                              [-chnl[1], chnl[0], 0]])

                R = np.eye(3) + np.sin(rotation)*K + (1 - np.cos(rotation))*(K @ K)

                flattened = waves.transpose(1, 0, 2).reshape(3, -1)

                rotated = (R@flattened).reshape(3, nsta, ndp).transpose(1, 0, 2)

                waves_aug = rotated

            else:
                self.logger.error("mode must be '2d' or '3d'")
                raise ValueError("mode must be '2d' or '3d'")
            
        return waves_aug, pickarr, stnout, []