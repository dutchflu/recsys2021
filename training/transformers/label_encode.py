"""
sklearn lable encode
"""
import pandas as pd
from sklearn import preprocessing
from training.transformers.transformer import Transformer
from commons.log import log


class LabelEncode(Transformer):
    """
    label encode
    """
    def __init__(self, cols):
        super().__init__(cols)
        self.label_encoders = {}

    def fit(self, data: pd.DataFrame):
        log.info("LabelEncode fit: %s", self.cols)
        for col in self.cols:
            self.label_encoders[col] = preprocessing.LabelEncoder()
            self.label_encoders[col] = self.label_encoders[col].fit(data[col])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info("LabelEncode transform: %s", self.cols)
        for col in self.cols:
            data.loc[~data[col].isin(self.label_encoders[col].classes_), col] = -1
            data.loc[data[col].isin(self.label_encoders[col].classes_), col] = self.label_encoders[col].transform(data[col][data[col].isin(self.label_encoders[col].classes_)])
            # data[col] = self.label_encoders[col].transform(data[col])
        return data
