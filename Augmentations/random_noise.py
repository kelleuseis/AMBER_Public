import numpy as np
from itertools import chain

from registry import augmentation_registry
from Augmentations.base_augmentation import BaseAugmentation, AugmentationRequest
from Augmentations.utils import normalise_addwave

    
    
@augmentation_registry.register("random_noise")
class RandomNoise(BaseAugmentation):
    '''
    Adds extra noise from other/same dataset(s). Can be specified
    to match original stations. 
    
    Inspired by following publication:
    
        Jun Zhu, Lihua Fang, Fajun Miao, Liping Fan, Ji Zhang,
        Zefeng Li (2024).
        Deep learning and transfer learning of earthquake and 
        quarry-blast discrimination: applications to southern 
        California and eastern Kentucky 
        Geophysical Journal International (236)

        https://doi.org/10.1093/gji/ggad463
        
    Parameters
    --------------------------------------------
    same_dataset: bool
        Use events from the same dataset for overlapping
        
    same_stations: bool
        Match station indices when overlapping (Unused
        when same_dataset is False)
        
    noise_amp_min: float
        Minimum scaling factor on standardized noise trace 
        relative to original signal standard deviation per
        station
        
    noise_amp_max: float
        Maximum scaling factor on standardized noise trace 
        relative to original signal standard deviation per
        station
    '''
    required_params = ["same_dataset", "same_stations"]
    optional_params = {"noise_amp_min":0.5, "noise_amp_max":1.0}
    scope = "windowed"

    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        if np.random.random() < self.augment_chance:
            return waves, pickarr, stnout, [
                NoiseAddRequest(
                    same_dataset=self.same_dataset,
                    same_stations=self.same_stations,
                    noise_amp_min=self.noise_amp_min, 
                    noise_amp_max=self.noise_amp_max
                )
            ]

        return waves, pickarr, stnout, []

    
    
class NoiseAddRequest(AugmentationRequest):
    def __init__(self, same_dataset, same_stations, noise_amp_min, noise_amp_max):
        self.same_dataset = same_dataset
        self.same_stations = same_stations
        self.noise_amp_min = noise_amp_min
        self.noise_amp_max = noise_amp_max

    def apply(self, context, waves, pickarr):
        if self.same_stations:
            stnout_temp = context.stnout
        else:
            stnout_temp = None
            
        waves_aug = waves.copy()
            
        if self.same_dataset:
            ds = context.eventdf['dataset'].iloc[0]
            id_list = context.dataset.noise_ids_dataset.get(ds, [])

        else:
            id_list = list(chain.from_iterable(context.dataset.noise_ids_dataset.values()))

        if len(id_list) == 0:
            context.dataset.logger.debug("Skipping augmentation: no sample candidates")
            
        else:
            temp_id = np.random.choice(id_list)
            
            waves_temp, eventdf_temp = context.dataset.extract_sample(
                temp_id, context.trace_grp
            )
            waves_temp, _, _, _ = context.dataset.extract_window(
                waves_temp, eventdf_temp, stnout_temp
            )
            
            waves_temp = normalise_addwave(waves, waves_temp, self.noise_amp_min, self.noise_amp_max)
            waves_aug += waves_temp
            
        return waves_aug, pickarr