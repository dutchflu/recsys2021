"""
data provisions
"""
from typing import Dict, Any, List
import json
import pandas as pd
import dask.dataframe as dd
from pydantic import BaseModel

from commons.config_factory import ConfigFactory
from commons.log import log


class LoadConfig(BaseModel):
    """Configuration to specify which data is loaded"""
    raw_data_path: str


class DataConfig(BaseModel):
    """Class to describe the data

    Used in models for feature specific hyperparameters."""
    label_cols: List[str]
    custom_features: List[str]
    categorical_features: str


class DataProviderConfig(BaseModel):
    """Holds all configuration a DataProvider needs"""
    load_config: LoadConfig
    data_config: DataConfig


def batch_read_dask(raw_data_path):
    """raw_data_path can contain wildcard
    e.g. "data/part-*.csv"
    """
    log.info("Dask reading")
    df: pd.DataFrame = dd.read_csv(raw_data_path,
                                   assume_missing=True)
    log.info("Converting to pandas dataframe")
    df_pd = df.compute().reset_index(drop=True)
    return df_pd


def load_config(config_path: str, config_factory: ConfigFactory):
    with open(config_path) as f:
        config_dict: Dict[str, Any] = json.load(f)
        pipeline_type = config_dict["pipeline_type"]
        return config_factory.from_config(pipeline_type, config_dict)
