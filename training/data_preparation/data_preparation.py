"""
data preparation
"""
from commons.data_provision import DataProviderConfig, batch_read_dask, DataConfig
from commons.data_cleaning import ColumnSelector
from commons.log import log

class DataProvider:
    """
    class for data loading
    """

    def __init__(self, config: DataProviderConfig):
        self.config = config
        self._column_selector = ColumnSelector(config.data_config)

    def _read_data(self, raw_data_path):
        raw_data = batch_read_dask(raw_data_path)

        return raw_data

    def _process_data(self, data):
        log.info("data shape before column selection: %s", data.shape)
        data = self._column_selector.transform(data)
        log.info("data shape after column selection: %s", data.shape)

        return data

    def get_data(self):
        data = self._read_data(self.config.load_config.raw_data_path)
        data = self._process_data(data)

        return data

    def get_data_config(self) -> DataConfig:
        """return data config
        """
        return self.config.data_config
