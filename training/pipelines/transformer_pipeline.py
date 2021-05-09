"""
Transformer Pipeline
"""
import pandas as pd

from training.transformers.string_to_bool import StringToBool
from training.transformers.label_encode import LabelEncode

TRANSFORM_MAP = {
    "string_to_bool": ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"],
    "label_encode": ["engaged_with_user_id", "enaging_user_id", "language", "tweet_type"]
}


class TransformerPipeline:
    """Pipeline for feature transformation
    """
    def __init__(self):
        self._transformers = [
            StringToBool(TRANSFORM_MAP["string_to_bool"]),
            LabelEncode(TRANSFORM_MAP["label_encode"])
        ]

    def fit(self, data: pd.DataFrame):
        for transformer in self._transformers:
            transformer.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for transformer in self._transformers:
            data = transformer.transform(data)
        return data
