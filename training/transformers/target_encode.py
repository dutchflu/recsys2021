"""
target_encode
"""
import pandas as pd
from category_encoders import TargetEncoder
from training.transformers.transformer import Transformer
from commons.log import log


class TargetEncode(Transformer):
    """
    target encode
    """
    def __init__(self, cols, targets):
        super().__init__(cols)
        self.targets = targets
        self.encoders = {}

    def fit(self, data: pd.DataFrame):
        log.info("TargetEncode fit: %s", self.targets)
        for target in self.targets:
            self.encoders["enc_{}".format(target)] = TargetEncoder(cols=self.cols,
                                                                   handle_missing="return_nan")
            log.info("Target encoding fit for target: %s", target)
            self.encoders["enc_{}".format(target)].fit(data[self.cols], data[target])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info("TargetEncode transform: %s", self.targets)
        for target in self.targets:
            encoded_features_values = self.encoders["enc_{}".format(target)].transform(
                data[self.cols], data[target])
            encoded_features = {}
            for encoded_feature in encoded_features_values:
                encoded_features["{}_{}".format(encoded_feature, target)] = encoded_features_values[encoded_feature]
            data = data.assign(**encoded_features)
        return data
