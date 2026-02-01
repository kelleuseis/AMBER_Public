import argparse
import importlib
import os

from dataclasses import dataclass, field
from typing import List, Dict, Any

from registry import labeller_registry


@dataclass
class LabellerConfig:
    dynamic_params: Dict[str, Any] = field(default_factory=dict)
        
        
        
def create_labeller(lbl_name, lbl_cfg: LabellerConfig):
    return labeller_registry.get(lbl_name)(lbl_cfg)



# Automatically import Python files in the labeller directory
labeller_dir = os.path.dirname(__file__)
for file in os.listdir(labeller_dir):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[:-3]
        importlib.import_module(f"Labeller.{module_name}")