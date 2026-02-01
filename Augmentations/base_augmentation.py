import numpy as np
import random, logging, abc

from registry import augmentation_registry


class AugmentationRequest:
    def apply(self, context, waves, pickarr):
        raise NotImplementedError


def initialize_params(instance, param_dict, required_params, optional_params):
    '''For flexible parameter setting (extra params will be ignored)'''
    for param in required_params:
        if param not in param_dict:
            raise ValueError(f"Missing required parameter: {param}")
        setattr(instance, param, param_dict[param])

    for param, default in optional_params.items():
        setattr(instance, param, param_dict.get(param, default))
        

class BaseAugmentation(abc.ABC):
    '''
    Base augmentation class
    
    Augmentations that require raw data -> scope="raw"
    Augmentations that may use windowed data -> scope="windowed"
    Augmentations that require addtional data from other events -> scope="windowed" + AugmentationRequest
    
    
    Base Parameters
    -----------------------------------
    augment_chance: float
        probability [0, 1] of applying augmentation
        
    log_level: str
        logging level ("ERROR", "WARNING", "INFO", "DEBUG")
    '''
    required_params = []
    optional_params = {"augment_chance":0.4, "log_level":"INFO"}    
    scope: str = None    # "raw" or "windowed"
        
    def __init_subclass__(cls):
        if cls.scope not in ("raw", "windowed"):
            raise TypeError(
                f"{cls.__name__} must define scope = 'raw' or 'windowed'"
            )
            
    def __init__(self, param_dict):
        merged_optional_params = {
            **BaseAugmentation.optional_params,
            **self.optional_params
        }
        initialize_params(self, param_dict, self.required_params, merged_optional_params)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)  
    
    
    def augment_raw(self, waves_all, eventdf, samplerate=None):
        '''
        For augmentations that work on raw data
        
        Parameters
        ---------------------------------
        waves_all: np.array
            raw trace data
            (n_stations, n_chnl, n_datapoints)
            
        eventdf: pandas.DataFrame
            raw trace metadata (not to be mutated!)
        
        Outputs
        ---------------------------------
        waves_all: np.array
            augmented raw trace data
            (n_stations, n_chnl, n_datapoints)
            
        samplerate: int
            sampling rate in Hz
        '''
        raise NotImplementedError

        
    def augment_windowed(self, waves, pickarr, stnout):
        '''
        For augmentations that work on windowed data
        
        Parameters
        ---------------------------------
        waves: np.array
            windowed trace data
            (n_stations, n_chnl, n_datapoints)
            
        pickarr: np.array
            pick times in datapoints
            (n_stations, 2(P,S), n_events)
            
        stnout: 1D np.array
            list of station indices
        
        Outputs
        ---------------------------------
        waves: np.array
            augmented windowed trace data
            (n_stations, n_chnl, n_datapoints)
            
        pickarr: np.array
            augmented pick times in datapoints 
            
        stnout: 1D np.array
            augmented list of station indices
            
        augrequests: list of Augmentations.base_augmentation.AugmentationRequest
            augmentation requests for PyTorch dataset to run
            (when requiring data from other events)
        '''
        raise NotImplementedError
