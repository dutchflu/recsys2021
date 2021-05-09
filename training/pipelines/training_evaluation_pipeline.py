"""
training evaluation pipeline
"""
from pydantic import BaseModel, Extra
import pandas as pd

from commons.data_provision import DataProviderConfig
from commons.log import log
from training.models.model import ModelConfig
from training.data_preparation.data_preparation import DataProvider
from training.pipelines.splitter import Splitter, Split

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
    def __init__(self, data_provider: DataProvider,
                 splitter: Splitter):
        self.data_provider: DataProvider = data_provider
        self.splitter = splitter

    def run(self):
        all_splits = []
        data = self.data_provider.get_data()
        split_num = 0
        for split in self.splitter.split(data, self.data_provider.config.load_config):
            results = self._run_split(data, split, split_num)
            all_splits.append(results)
            split_num += 1

    def _run_split(self, data: pd.DataFrame, split: Split, split_num: int):
        train_set, test_set = split.data(data)
        log.info("train_set shape: %s", train_set.shape)
        log.info("test_set shape: %s", test_set.shape)
        # training_pipeline = self._run_training_pipeline(train_set)

        return {}
