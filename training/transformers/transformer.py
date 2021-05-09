"""Module for the Transformer"""
from abc import abstractmethod
import pandas as pd


class Transformer:
    """Class to subclass for Transformers in a TransformerPipeline"""

    def __init__(self, cols):
        self.cols = cols

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
