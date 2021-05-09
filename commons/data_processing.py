"""
data processing
"""
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import ShuffleSplit


class TrainValidationSplit(BaseModel):
    """Container for two subsets of the same DataFrame"""
    class Config:
        """Allow DataFrames"""
        arbitrary_types_allowed = True

    train_data: pd.DataFrame
    validation_data: pd.DataFrame

    def __iter__(self):
        """Iterator for tuple unpacking"""
        return iter([self.train_data, self.validation_data])


def split_train_val(raw_data: pd.DataFrame, val_size: float, random_state=42, return_index=False):
    """Split a DataFrame of size N into two MECE DataFrames of val_size*N and (1-val_size)*N

    Use random_state to keep results consistent or random, default to consistent"""
    gss = ShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

    train_inds, val_inds = next(gss.split(raw_data))

    if return_index:
        return raw_data.index[train_inds], raw_data.index[val_inds]
    else:
        val_data = raw_data.iloc[val_inds].reset_index(drop=True)
        train_data = raw_data.iloc[train_inds].reset_index(drop=True)

        return TrainValidationSplit(train_data=train_data, validation_data=val_data)
