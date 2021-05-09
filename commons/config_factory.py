"""
config factory
"""
from typing import Dict, Type
from pydantic import BaseModel


class ConfigFactory:
    """Registers and initialises configuration dataclasses with pydantic.BaseModel"""
    config_classes: Dict[str, Type[BaseModel]] = {}

    def from_config(self, config_type: str, config_dictionary: Dict[str, Dict]) -> BaseModel:
        config_class = self.config_classes[config_type]
        return config_class.parse_obj(config_dictionary)

    def register(self, name: str, config_class: Type[BaseModel]):
        self.config_classes.update({name: config_class})
