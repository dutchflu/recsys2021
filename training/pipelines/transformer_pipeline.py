"""
Transformer Pipeline
"""
import pandas as pd

from training.transformers.string_to_bool import StringToBool
from training.transformers.label_encode import LabelEncode
from training.transformers.target_encode import TargetEncode

TRANSFORM_MAP = {
    "label_cols": ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"],
    "categorical_features": ["engaged_with_user_id", "enaging_user_id", "language", "tweet_type"],
    "normal_features": ["enaging_user_follower_count"]
}


class TransformerPipeline:
    """Pipeline for feature transformation
    """
    def __init__(self, data_config):
        self.data_config = data_config
        self._transformers = [
            StringToBool(data_config.label_cols),
            LabelEncode(data_config.categorical_features),
            TargetEncode(data_config.categorical_features, data_config.label_cols)
        ]

    def fit(self, data: pd.DataFrame):
        for transformer in self._transformers:
            transformer.fit(data)
            data = transformer.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for transformer in self._transformers:
            data = transformer.transform(data)
        return data

    def get_updated_features(self):
        categorical_features = ["{}_{}".format(i, j) for i in self.data_config.categorical_features for j in self.data_config.label_cols]
        features = self.data_config.normal_features + categorical_features
        return features
