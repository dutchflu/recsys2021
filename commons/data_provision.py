"""
data provisions
"""
from typing import Dict, Any, List
import json
from pydantic import BaseModel

from commons.config_factory import ConfigFactory


class LoadConfig(BaseModel):
    """Configuration to specify which data is loaded"""
    raw_data_path: str


class DataConfig(BaseModel):
    """Class to describe the data

    Used in models for feature specific hyperparameters."""
    custom_features: List[str]
    categorical_features: str


class DataProviderConfig(BaseModel):
    """Holds all configuration a DataProvider needs"""
    load_config: LoadConfig
    data_config: DataConfig


def load_config(config_path: str, config_factory: ConfigFactory):
    with open(config_path) as f:
        config_dict: Dict[str, Any] = json.load(f)
        pipeline_type = config_dict["pipeline_type"]
        return config_factory.from_config(pipeline_type, config_dict)
