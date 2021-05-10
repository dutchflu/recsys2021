"""
lightgbm model
"""
import lightgbm as lgb
import pandas as pd
from training.models.model import Model, ModelConfig
from commons.log import log


class LightGbmModel(Model):
    """Lightgbm model"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = lgb.LGBMClassifier(**config.hyperparameters.dict())
        # self._model = None
        # self.params = config.hyperparameters.dict()

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, features: list, labels: list):
        log.info("training")
        eval_set = [(val_data[features], val_data[labels[0]])]
        self._model.fit(X=train_data[features],
                        y=train_data[labels[0]],
                        eval_set=eval_set,
                        eval_names=['val'],
                        feature_name=features,
                        categorical_feature='auto',
                        **self._fit_config)

        # d_train = lgb.Dataset(train_data[features], label=train_data[labels])
        # self._model = lgb.train(self.params, d_train, 10)

    def predict(self, data: pd.DataFrame, features: list):
        log.info("predicting")
        ypred = self._model.predict(data[features])
        return ypred
