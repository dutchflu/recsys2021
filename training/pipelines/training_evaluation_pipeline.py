"""
training evaluation pipeline
"""
from pydantic import BaseModel, Extra

from commons.data_provision import DataProviderConfig
from commons.log import log
from training.models.model import ModelConfig
from training.data_preparation.data_preparation import DataProvider


class PipelineConfig(BaseModel):
    """Extra setup for a pipeline, like TrainConfig or EvaluateConfig"""
    class Config:
        """different per pipeline"""
        extra = Extra.allow


class TrainConfig(PipelineConfig):
    """Configuration for training pipeline"""
    val_size: float = 0


class TrainingEvaluationPipelineConfig(BaseModel):
    """Configuration for the TrainingEvaluationPipeline class"""
    pipeline_type: str
    data_provider_config: DataProviderConfig
    model_config: ModelConfig
    train_config: TrainConfig


class TrainingEvaluationPipeline:
    """
    Train and evaluate
    """
    def __init__(self, data_provider: DataProvider):
        self.data_provider: DataProvider = data_provider

    def run(self):
        data = self.data_provider.get_data()
        log.info(data.columns)
