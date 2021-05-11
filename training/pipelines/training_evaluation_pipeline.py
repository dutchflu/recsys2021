"""
training evaluation pipeline
"""
from pydantic import BaseModel, Extra
import pandas as pd
import numpy as np

from commons.data_provision import DataProviderConfig
from commons.log import log
from commons.data_processing import split_train_val
from training.models.model import ModelConfig
from training.data_preparation.data_preparation import DataProvider
from training.pipelines.splitter import Splitter, Split
from training.pipelines.transformer_pipeline import TransformerPipeline
from training.models.model import Model
from training.models.lightgbm import LightGbmModel
from training.evaluation.evaluator import RceEvaluator, AvgPrecisionEvaluator

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
                 model: Model,
                 pipeline_config: TrainConfig):
        self.transformer_pipeline = transformer_pipeline
        self.model = model
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
        log.info("transforming training data")
        self.train_data = self.transformer_pipeline.fit_transform(self.train_data)
        # self.train_data = self.transformer_pipeline.transform(self.train_data)
        log.info("cols after data transform: %s", self.train_data.columns)
        log.info("transforming validation data")
        self.validation_data = self.transformer_pipeline.transform(self.validation_data)
        return {"train_data": self.train_data, "val_data": self.validation_data, "transformer_pipeline": self.transformer_pipeline}

    def _train(self):
        self.model.train(self.train_data, self.validation_data,
                         self.transformer_pipeline.get_updated_features(),
                         self.transformer_pipeline.get_labels())
        return self.model

    def get_transformer_pipeline(self):
        return self.transformer_pipeline

    def get_model(self):
        return self.model

    def get_pipeline_config(self) -> TrainConfig:
        return self.pipeline_config

    def get_train_val_data(self):
        return self.train_data, self.validation_data

    def run(self):
        self._generate_train_val_data()
        self._transform_data()
        log.info(self.transformer_pipeline.get_updated_features())
        self._train()


class PredictionPipeline:
    """Prediction Pipeline"""
    def __init__(self,
                 test_data: pd.DataFrame,
                 transformer_pipeline: TransformerPipeline,
                 model: Model):
        self.transformer_pipeline = transformer_pipeline
        self.model = model
        self.test_data = test_data

    def transform_data(self):
        self.test_data = self.transformer_pipeline.transform(self.test_data)
        return self.test_data

    def predict(self) -> np.ndarray:
        ypred = self.model.predict(self.test_data, self.transformer_pipeline.get_updated_features())
        return ypred

    def run(self) -> np.ndarray:
        self.transform_data()
        ypred = self.predict()
        label_cols = self.transformer_pipeline.get_labels()
        return ypred, self.test_data[label_cols]


class TrainingEvaluationPipeline:
    """
    Train and evaluate
    """
    def __init__(self, data_provider: DataProvider,
                 splitter: Splitter,
                 model_config: ModelConfig,
                 pipeline_config: TrainConfig):
        self.data_provider: DataProvider = data_provider
        self.splitter = splitter
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.evaluators = {"rce": RceEvaluator(), "ap": AvgPrecisionEvaluator()}

    def run(self):
        results_all_splits = []
        data = self.data_provider.get_data()
        split_num = 0
        for split in self.splitter.split(data, self.data_provider.config.load_config):
            results = self._run_split(data, split, split_num)
            results_all_splits.append(results)
            split_num += 1
        log.info("results for all split: %s", results_all_splits)

    def _run_split(self, data: pd.DataFrame, split: Split, split_num: int):
        train_set, test_set = split.data(data)
        log.info("train_set shape: %s", train_set.shape)
        log.info("test_set shape: %s", test_set.shape)
        training_pipeline = self._run_training_pipeline(train_set)
        transformer_pipeline = training_pipeline.get_transformer_pipeline()
        model = training_pipeline.get_model()
        ypred, ytrue = self._run_prediction_pipeline(test_set, transformer_pipeline, model)
        scores_for_split = self._score(ypred, ytrue)
        log.info("results for split %s: %s", split_num, scores_for_split)
        return scores_for_split

    def _training_pipeline_factory(self, train_set):
        transformer_pipeline = TransformerPipeline(self.data_provider.get_data_config())
        model = LightGbmModel(self.model_config)
        return TrainingPipeline(train_set, transformer_pipeline, model, self.pipeline_config)

    def _prediction_pipeline_factory(self, test_set, transformer_pipeline, model):
        return PredictionPipeline(test_set, transformer_pipeline, model)

    def _run_training_pipeline(self, train_set):
        training_pipeline = self._training_pipeline_factory(train_set)
        training_pipeline.run()
        return training_pipeline

    def _run_prediction_pipeline(self, test_set: pd.DataFrame, transformer_pipeline, model):
        prediction_pipeline = self._prediction_pipeline_factory(test_set, transformer_pipeline, model)
        ypred, ytrue = prediction_pipeline.run()
        return ypred, ytrue

    def _score(self, ypred, ytrue):
        scores = {}
        for name, evaluator in self.evaluators.items():
            score = evaluator.score(ypred, ytrue)
            scores[name] = score
        return scores
