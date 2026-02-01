import argparse
import importlib
import os

from dataclasses import dataclass, field
from typing import List, Dict, Any

from registry import augmentation_registry


@dataclass
class AugmentationConfig:
    augmentations: List[Dict[str, Any]] = field(default_factory=list)


def load_augmentations(config: AugmentationConfig):
    augmentations = []
    for aug_conf in config.augmentations:
        aug_type = aug_conf["type"]
        aug_params = aug_conf.get("params", {})
        augmentation_cls = augmentation_registry.get(aug_type)
        augmentations.append(augmentation_cls(aug_params))
    return augmentations
       
    
# Automatically import Python files in the augmentation directory
augmentations_dir = os.path.dirname(__file__)
for file in os.listdir(augmentations_dir):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[:-3]
        importlib.import_module(f"Augmentations.{module_name}")