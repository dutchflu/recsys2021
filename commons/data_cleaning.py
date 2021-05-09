"""
data cleaning
"""
import pandas as pd

from commons.data_provision import DataConfig
from commons.log import log


class ColumnSelector:
    """Filter columns needed for a task
    """

    def __init__(self, data_config: DataConfig):
        self._config = data_config

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_config = self._config
        cols = data_config.custom_features + data_config.label_cols
        log.info("Column Selection. cols: %s", cols)
        data = data[list(set(cols))]
        return data
