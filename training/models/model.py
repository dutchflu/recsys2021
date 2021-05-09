"""
Model Basics
"""
from pydantic import BaseModel, Extra


class Hyperparameters(BaseModel):
    """Hyperparameters to use in a Model"""
    class Config:
        """allows the Hyperparameters to read any attributes."""
        extra = Extra.allow


class ModelConfig(BaseModel):
    """Model Config"""
    model_type: str
    hyperparameters: Hyperparameters
