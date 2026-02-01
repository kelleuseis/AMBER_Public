import numpy as np

from registry import augmentation_registry
from Augmentations.base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("random_phase_rotation")
class RandomPhaseRotation(BaseAugmentation):
    '''
    Randomly apply same constant phase rotation to 
    all stations
    '''
    scope = "windowed"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        if np.random.random() < self.augment_chance:
            fft_waves = np.fft.rfft(waves, axis=-1)

            phi = np.random.rand() * 2 * np.pi
            phase_rotation = np.exp(1j * phi)

            rotated_fft = fft_waves * phase_rotation

            waves = np.fft.irfft(
                rotated_fft, n=waves.shape[-1], axis=-1
            )
            
        return waves, pickarr, stnout, []
