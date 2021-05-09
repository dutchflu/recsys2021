"""
config factory used for training
"""
from typing import Type
from pydantic import BaseModel

from commons.config_factory import ConfigFactory
from training.pipelines.training_evaluation_pipeline import TrainingEvaluationPipelineConfig

config_factory = ConfigFactory()
config_class: Type[BaseModel]
for name, config_class in (
        ("training_evaluation_pipeline", TrainingEvaluationPipelineConfig),
):
    config_factory.register(name, config_class)
