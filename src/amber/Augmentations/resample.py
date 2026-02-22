import numpy as np
from scipy.signal import resample_poly

from amber.registry import augmentation_registry
from .base_augmentation import BaseAugmentation, AugmentationRequest

    
    
@augmentation_registry.register("resample")
class Resample(BaseAugmentation):
    '''
    Resample all traces to a new sample rate.
    
    Parameters
    ---------------------------------
    samplerate: float
        target sampling rate in Hz
        
    windowlength: int
        expected window length for model input
    '''
    optional_params = {"samplerate":4000, "windowlength":4000}
    scope = "raw"
    
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_raw(self, waves_all, eventdf, samplerate):
        if samplerate is None:
            raw_samplerate = eventdf['trace_sampling_rate_hz'].iloc[0]
        else:
            raw_samplerate = samplerate

        ndp = waves_all.shape[-1]
        new_wintime = self.windowlength / self.samplerate
        raw_wintime = ndp / raw_samplerate

        if raw_wintime <= new_wintime:
            min_newsamplerate = self.windowlength*raw_samplerate / ndp
            k = int(np.ceil(min_newsamplerate / self.samplerate))
            samplerate = k * self.samplerate
        else:
            samplerate = self.samplerate

        scale = samplerate / raw_samplerate
        waves_all_aug = resample_poly(
            waves_all, int(round(scale*1000)), 1000, axis=-1
        )

        return waves_all_aug, samplerate