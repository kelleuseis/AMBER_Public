'''
Defines registries used across modules to enable Hydra/YAML-driven configuration
and dynamic object instantiation. See extract.py for usage examples.
'''
class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"{name} is already registered.")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name):
        if name not in self._registry:
            raise ValueError(f"{name} is not registered.")
        return self._registry[name]

augmentation_registry = Registry()
labeller_registry = Registry()
