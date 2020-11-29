from sklearn.dummy import DummyClassifier
from bac.models.model_base import ModelBase
import pandas as pd
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DummyModel(ModelBase):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model == DummyClassifier(**kwargs)
        self.params == kwargs
        
        
    def do_fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train model. Extends model_base abstract class.
        Can be used to fit a Majority Class or Random Model

        Args:
            X_train (pd.DataFrame): feature set (user-day)
            y_train (pd.Series): targets. required, but not used.
            **fit_args:  kwargs for DummyClassifier.fit() method
            -- e.g., strategy="constant", constant=1
        """
        logging.info("\nTraining Dummy Model...")
        logger.info(f"Model Params: \n{self.params}")
        self.model.fit(X_train, y_train)
        self.fitted = True
        
    def save_training_info(self, X_train: pd.DataFrame):
        """Save columns used in training as instance variables

        Args:
            X_train (pd.DataFrame): feature set for training
        """
        assert self.fitted
        self.columns = X_train.columns.tolist()
        self.n_userdays = len(X_train)
    
    
    def do_predict(self, X_eval: pd.DataFrame) -> pd.Series:
        """ 
        Return probability scores [0, 1] for each row of X.
        First checks that columns match self.columns
        """
        scores = self.model.predict_proba(X_eval.values)
        if len(scores.shape) == 2:
            scores = scores[:, 1]
        return pd.Series(scores, index=X_eval.index)
        