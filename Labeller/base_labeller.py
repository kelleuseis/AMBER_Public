import logging, abc

from Labeller import LabellerConfig



def initialize_params(instance, config: LabellerConfig, required_params, optional_params):
    for param in required_params:
        if param not in config.dynamic_params:
            raise ValueError(f"Missing required parameter: {param}")
        setattr(instance, param, config.dynamic_params[param])

    for param, default in optional_params.items():
        setattr(instance, param, config.dynamic_params.get(param, default))


class BaseLabeller(abc.ABC):
    required_params = []
    optional_params = {"log_level": "INFO"}
    
    def __init__(self, config: LabellerConfig):
        merged_optional_params = {
            **BaseLabeller.optional_params,
            **self.optional_params
        }        
        initialize_params(self, config, self.required_params, merged_optional_params)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)        
 
    def __call__(self, pickarr, eventdf, stnout, window_idx, samplerate):
        return self.forward(pickarr, eventdf, stnout, window_idx, samplerate)
        
    def forward(self, pickarr, eventdf, stnout, window_idx, samplerate):
        raise NotImplementedError