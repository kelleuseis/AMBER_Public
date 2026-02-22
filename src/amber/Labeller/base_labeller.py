import logging, abc

from amber.Labeller import LabellerConfig



def initialize_params(instance, config: LabellerConfig, required_params, optional_params):
    '''For flexible parameter setting (extra params will be ignored)'''
    for param in required_params:
        if param not in config.dynamic_params:
            raise ValueError(f"Missing required parameter: {param}")
        setattr(instance, param, config.dynamic_params[param])

    for param, default in optional_params.items():
        setattr(instance, param, config.dynamic_params.get(param, default))


class BaseLabeller(abc.ABC):
    '''
    Base augmentation class
    
    Augmentations that require raw data -> scope="raw"
    Augmentations that may use windowed data -> scope="windowed"
    Augmentations that require addtional data from other events -> scope="windowed" + AugmentationRequest
    
    
    Base Parameters
    -----------------------------------
    log_level: None or str
        logging level ("ERROR", "WARNING", "INFO", "DEBUG")
    '''
    required_params = []
    optional_params = {"log_level":None}
    
    def __init__(self, config: LabellerConfig):
        merged_optional_params = {
            **BaseLabeller.optional_params,
            **self.optional_params
        }        
        initialize_params(self, config, self.required_params, merged_optional_params)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if self.log_level is not None:
            level = getattr(logging, self.log_level.upper(), None)
            if level is not None:
                self.logger.setLevel(level)
 

    def __call__(self, pickarr, eventdf, stnout, window_idx, samplerate):
        return self.forward(pickarr, eventdf, stnout, window_idx, samplerate)
        
    def forward(self, pickarr, eventdf, stnout, window_idx, samplerate):
        raise NotImplementedError