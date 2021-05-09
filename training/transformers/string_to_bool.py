"""
convert string to bool
"""
import pandas as pd
from training.transformers.transformer import Transformer
from commons.log import log


def convert_str_to_bool(string_x):
    label = 0 if pd.isnull(string_x) else 1
    return label


class StringToBool(Transformer):
    """
    string to bool
    """
    def fit(self, data: pd.DataFrame):
        log.info("StringToBool fit: pass")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info("StringToBool transform: %s", self.cols)
        for col in self.cols:
            data[col] = data[col].map(convert_str_to_bool)
        return data
