from abc import abstractmethod

import pandas as pd

from bac.models.model_base import ModelBase


class ClassifierBase(ModelBase):
    @abstractmethod
    def do_fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train model

        Args:
            X (pd.DataFrame): features (user-day)
            y (pd.Series): prediction targets (BAC level>=0.08)
        """
        pass
    
    @abstractmethod
    def do_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Args:
            X (pd.DataFrame): [description]

        Returns:
            pd.Series: scores for each row of X - probability
        """
        pass