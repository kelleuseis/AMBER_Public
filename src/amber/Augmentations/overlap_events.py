import numpy as np
from itertools import chain

from amber.registry import augmentation_registry
from .base_augmentation import BaseAugmentation, AugmentationRequest
from .utils import normalise_addwave

    
    
@augmentation_registry.register("overlap_events")
class OverlapEvents(BaseAugmentation):
    '''
    Adds extra events from other/same dataset(s). Can be specified
    to match original stations. A event/phase-centric window is 
    randomly extracted from the addition traces to be superimposed.
    
    Inspired by following publication:
    
        Sun, H., Ross, Z. E., Zhu, W., & Azizzadenesheli, K. (2023).
        Phase neural operator for multi-station picking of seismic 
        arrivals
        Geophysical Research Letters, 50, e2023GL10643
        
        https://doi.org/10.1029/2023GL106434x
        
        
    Parameters
    --------------------------------------------
    same_dataset: bool
        Use events from the same dataset for overlapping
        
    same_stations: bool
        Match station indices when overlapping (Unused
        when same_dataset is False)
        
    event_amp_min: float
        Minimum scaling factor on standardized event trace 
        relative to original signal standard deviation per
        station
        
    event_amp_max: float
        Maximum scaling factor on standardized event trace 
        relative to original signal standard deviation per
        station
    '''
    required_params = ["same_dataset", "same_stations"]
    optional_params = {"event_amp_min":1, "event_amp_max":2}
    scope = "windowed"

    def __init__(self, param_dict):
        super().__init__(param_dict)

    def augment_windowed(self, waves, pickarr, stnout):
        if np.random.random() < self.augment_chance:
            return waves, pickarr, stnout, [
                EventAddRequest(
                    same_dataset=self.same_dataset,
                    same_stations=self.same_stations,
                    event_amp_min=self.event_amp_min, 
                    event_amp_max=self.event_amp_max
                )
            ]

        return waves, pickarr, stnout, []    
    
    
    
class EventAddRequest(AugmentationRequest):
    def __init__(self, same_dataset, same_stations, event_amp_min, event_amp_max):
        self.same_dataset = same_dataset
        self.same_stations = same_stations
        self.event_amp_min = event_amp_min
        self.event_amp_max = event_amp_max

    def apply(self, context, waves, pickarr):
        if self.same_stations:
            stnout_temp = context.stnout
        else:
            stnout_temp = None
            
        waves_aug = waves.copy()
            
        if self.same_dataset:
            ds = context.eventdf['dataset'].iloc[0]
            id_list = context.dataset.event_ids_dataset.get(ds, [])

        else:
            id_list = list(chain.from_iterable(context.dataset.event_ids_dataset.values()))

        if len(id_list) == 0:
            context.dataset.logger.debug("Skipping augmentation: no sample candidates")
            
        else:
            temp_id = np.random.choice(id_list)
            
            waves_temp, eventdf_temp = context.dataset.extract_sample(
                temp_id, context.trace_grp
            )
            waves_temp, pickarr_temp, _, _ = context.dataset.extract_window(
                waves_temp, eventdf_temp, stnout_temp
            )

            pickarr = np.concatenate((pickarr, pickarr_temp), axis=2)
            
            waves_temp = normalise_addwave(waves, waves_temp, self.event_amp_min, self.event_amp_max)
            waves_aug += waves_temp
            
        return waves_aug, pickarr