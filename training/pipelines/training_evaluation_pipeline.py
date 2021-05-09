"""
training evaluation pipeline
"""
from pydantic import BaseModel, Extra
import pandas as pd

from commons.data_provision import DataProviderConfig
from commons.log import log
from commons.data_processing import split_train_val
from training.models.model import ModelConfig
from training.data_preparation.data_preparation import DataProvider
from training.pipelines.splitter import Splitter, Split
from training.pipelines.transformer_pipeline import TransformerPipeline

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


class TrainingPipeline:
    """Training Pipeline"""
    def __init__(self,
                 train_data: pd.DataFrame,
                 transformer_pipeline: TransformerPipeline,
                 # model: Model,
                 pipeline_config: TrainConfig):
        self.transformer_pipeline = transformer_pipeline
        # self.model = model
        self.pipeline_config = pipeline_config
        self.train_data = train_data
        self.validation_data = None

    def _generate_train_val_data(self):
        all_data = self.train_data
        val_size = self.pipeline_config.val_size
        if val_size > 0:
            self.train_data, self.validation_data = split_train_val(all_data, val_size, return_index=False)
        else:
            self.validation_data = None
        return {"train_data": self.train_data, "val_data": self.validation_data}

    def _transform_data(self):
        log.info("cols before data transform: %s", self.train_data.columns)
        self.transformer_pipeline.fit(self.train_data)
        log.info("transforming training data")
        self.train_data = self.transformer_pipeline.transform(self.train_data)
        log.info("cols after data transform: %s", self.train_data.columns)
        log.info("transforming validation data")
        self.validation_data = self.transformer_pipeline.transform(self.validation_data)
        return {"train_data": self.train_data, "val_data": self.validation_data, "transformer_pipeline": self.transformer_pipeline}

    # def _train(self):
    #     self.model.train(self.train_data, self.validation_data,
    #                      self.transformer_pipeline.get_output_data_config(self.data_provider.get_data_config()))
    #     return self.model

    def get_transformer_pipeline(self):
        return self.transformer_pipeline

    # def get_model(self):
    #     return self.model

    def get_pipeline_config(self) -> TrainConfig:
        return self.pipeline_config

    def get_train_val_data(self):
        return self.train_data, self.validation_data

    def run(self):
        self._generate_train_val_data()
        self._transform_data()
        log.info(self.transformer_pipeline.get_updated_features())
        # self._train()


class TrainingEvaluationPipeline:
    """
    Train and evaluate
    """
    def __init__(self, data_provider: DataProvider,
                 splitter: Splitter,
                 pipeline_config: TrainConfig):
        self.data_provider: DataProvider = data_provider
        self.splitter = splitter
        self.transformer_pipeline = TransformerPipeline(self.data_provider.get_data_config())
        self.pipeline_config = pipeline_config

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
        training_pipeline = self._run_training_pipeline(train_set)

        return {}

    def _run_training_pipeline(self, train_set):
        training_pipeline = TrainingPipeline(train_set, self.transformer_pipeline, self.pipeline_config)
        training_pipeline.run()
        return training_pipeline
