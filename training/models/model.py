"""
Model Basics
"""
from typing import Dict, Any
from abc import abstractmethod
from pydantic import BaseModel, Extra
import pandas as pd
import numpy as np


class Hyperparameters(BaseModel):
    """Hyperparameters to use in a Model"""
    class Config:
        """allows the Hyperparameters to read any attributes."""
        extra = Extra.allow


class ModelConfig(BaseModel):
    """Model Config"""
    model_type: str
    hyperparameters: Hyperparameters
    fit_config: Dict[str, Any]


class Model:
    """
    Basic model class
    """
    def __init__(self, config: ModelConfig):
        self._fit_config = config.fit_config

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, features: list, labels: list):
        """train"""
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame, features: list) -> np.ndarray:
        """Predict for (new) data using the fit model"""
        ...
