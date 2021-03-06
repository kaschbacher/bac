import logging
import sys
from typing import Sequence, Dict
import pandas as pd
from pandas.api.types import is_numeric_dtype
import joblib

from abc import ABC, abstractmethod

from bac.features.feature_importances import compute_feature_importances

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ModelBase(ABC):
    def __init__(self):
        self.fitted = False,
        self.columns = []
        

    # def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    #     """Fit model by self.do_fit(), implemented in child class
    #     """
    #     self.do_fit(X_train, y_train)
    #     self.fitted = True
        
    # def predict(self, X_eval: pd.DataFrame) -> pd.Series:
    #     """Apply model to eval feature set by self.do_predict(),
    #     must be implemented in child class.
        
    #     Return a probability score [0, 1] for each row in X.
    #     """
    #     assert self.fitted
    #     assert self.columns == X_eval.columns
    #     return self.do_predict(X_eval)
    
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
    
    # def save_model(self, model_outpath: str) -> None:
    #     """
    #     Save serialized model to docker path.
    #     """
    #     joblib.dump(self, model_outpath)
    #     logger.info(f"Completed. Model saved: {model_outpath}")
        
    def do_save(self, model_folder: str, model_name: str):
        """Save a serialized model to a given folder

        Args:
            model_folder: location on the docker volume to save
            model_name: name of the serialized model file
        """
        model_outpath = f"{model_folder}/{model_name}.joblib"
        joblib.dump(self, model_outpath)
        logger.info(f"Completed. Model saved: {model_outpath}\n")