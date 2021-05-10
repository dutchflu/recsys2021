"""
Transformer Pipeline
"""
import pandas as pd

from training.transformers.label_transform import LabelTransform
from training.transformers.label_encode import LabelEncode
from training.transformers.target_encode import TargetEncode


class TransformerPipeline:
    """Pipeline for feature transformation
    """
    def __init__(self, data_config):
        self.data_config = data_config
        self._transformers = [
            LabelTransform(data_config.label_cols),
            LabelEncode(data_config.categorical_features),
            TargetEncode(data_config.categorical_features, data_config.label_cols)
        ]

    def fit_transform(self, data: pd.DataFrame):
        for transformer in self._transformers:
            transformer.fit(data)
            data = transformer.transform(data)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for transformer in self._transformers:
            data = transformer.transform(data)
        return data

    def get_updated_features(self):
        te_categorical_features = ["{}_{}".format(i, j) for i in self.data_config.categorical_features for j in self.data_config.label_cols]
        features = self.data_config.normal_features + te_categorical_features
        return features

    def get_labels(self):
        return self.data_config.label_cols
