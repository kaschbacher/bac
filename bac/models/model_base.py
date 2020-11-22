import logging
import sys
from typing import Sequence, Dict
import pandas as pd
from pandas import is_numeric_dtype
import joblib

from abc import ABC, abstractmethod

from bac.features.feature_importances import compute_feature_importances

logger = logging.getLogger(__name__)
logger.setLevel(stream=sys.stdout, level=logging.INFO)


class ModelBase(ABC):
    def __init__(
        self,
    ):
        self.fill_missing = False,# self.cleaned
        self.fitted = False,
        self.columns = []
        
    def fill_missing_data(self, 
        X: pd.DataFrame, x_cols: Sequence[str], missing_value: int=-999
    )->pd.DataFrame:
        """Fill in missing data with a given value

        Args:
            X (pd.DataFrame): Feature dataset X
            x_cols (Sequence[str]): Columns to apply fill

        Returns:
            pd.DataFrame: A df with missing data filled.
        """
        X_out = X
        if not self.fill_missing:
            X_out = X_out[x_cols].fillna(value=missing_value, axis=0)
            self.fill_missing = True
        return X_out

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit model by self.make_fit(), implemented in child class
        """
        if not self.fill_missing:
            self.fill_missing_data(X_train)
            #TODO: include x_cols
        self.do_fit(X_train, y_train)
        self.fitted = True
        
    def predict(self, X_eval: pd.DataFrame) -> pd.Series:
        """Apply model to eval feature set by self.do_predict(),
        must be implemented in child class.
        
        Return a probability score [0, 1] for each row in X.
        """
        assert self.fitted
        assert self.columns == X_eval.columns
        return self.do_predict(X_eval)
    
    @abstractmethod
    def do_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit model.  Implemented by child class.
        
        Args:
            X_train: Feature dataset to train on
            y_train: target (per user-day)
        """
        pass
    
    
    @abstractmethod
    def do_predict(self, X_eval: pd.DataFrame) -> pd.Series:
        """Return a probability score [0,1] for each row in X
        """
        pass
    
    def save_model(self, model_outpath: str) -> None:
        """
        Save serialized model to docker path.
        """
        joblib.dump(self, model_outpath)
        logger.info(f"Completed. Model saved to docker: {model_outpath}")