import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from Labeller.base_labeller import BaseLabeller
from registry import labeller_registry


@labeller_registry.register("tapered_labeller")
class TaperedLabeller(BaseLabeller):
    '''
    Parameters
    ----------------------------------
    dropoff: int
        
    windowlength: int
    '''
    required_params = ["dropoff", "windowlength"]
    
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, pickarr, eventdf, stnout, window_idx, samplerate):
        t = np.arange(self.windowlength)[None, None, None, :]
        picks = pickarr[..., None]

        valid = (~np.isnan(picks)) & (picks >= 0) & (picks < self.windowlength)

        dist = np.abs(t - picks)
        taper = 1 - dist / self.dropoff
        taper = np.where((dist <= self.dropoff) & valid, taper, 0.0)

        labels = taper.max(axis=2)

        noise_chnl = 1 - labels.sum(axis=1, keepdims=True)
        noise_chnl = np.clip(noise_chnl, 0.0, 1.0)

        labels = np.concatenate((labels, noise_chnl), axis=1)

        return [labels]