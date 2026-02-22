import numpy as np
from scipy.signal import resample_poly

from amber.registry import augmentation_registry
from .base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("random_resampling")
class RandomResampling(BaseAugmentation):
    '''
    Randomly resample all traces to a new sample rate.
    
    Parameters
    ---------------------------------
    maxrange: float
        lower bound of the multiplicative time-scaling 
        factor applied during random resampling
    
    minrange: float
        upper bound of the multiplicative time-scaling 
        factor applied during random resampling
    '''
    optional_params = {"maxrange":1.5, "minrange":0.9}
    scope = "raw"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_raw(self, waves_all, eventdf, samplerate):            
        if np.random.random() < self.augment_chance:
            if samplerate is None:
                raw_samplerate = eventdf['trace_sampling_rate_hz'].iloc[0]
            else:
                raw_samplerate = samplerate

            scale = np.random.uniform(self.minrange, self.maxrange)
            samplerate = raw_samplerate * scale

            waves_all_aug = resample_poly(
                waves_all, int(round(scale*1000)), 1000, axis=-1
            )
            
            return waves_all_aug, samplerate
        
        return waves_all, samplerate