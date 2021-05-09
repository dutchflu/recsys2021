"""
training evaluation pipeline
"""
from pydantic import BaseModel, Extra

from commons.data_provision import DataProviderConfig
from training.models.model import ModelConfig


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
