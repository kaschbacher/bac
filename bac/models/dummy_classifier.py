from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from bac.models.model_base import ModelBase
import pandas as pd
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DummyModel(ModelBase):
    
    def __init__(self, **kwargs):
        """Majority Class kwargs: strategy="constant", constant=1
        """
        super().__init__()
        self.model = DummyClassifier(**kwargs)
        self.params = kwargs
        
        
    def do_fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train model. Extends model_base abstract class.
        Can be used to fit a Majority Class or Random Model

        Args:
            X_train (pd.DataFrame): feature set (user-day)
            y_train (pd.Series): targets. required, but not used.
            **fit_args:  kwargs for DummyClassifier.fit() method.
            -- not used here, just for compatibility
            
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
    
    
    def do_predict(self, X: pd.DataFrame) -> pd.Series:
        """ 
        Return probability scores [0, 1] for each row of X.
        First checks that columns match self.columns
        """
        scores = self.model.predict_proba(X.values)
        if len(scores.shape) == 2:
            scores = scores[:, 1]
        return pd.Series(scores, index=X.index)
        
    
    def get_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Enable logging of train/eval auc during training,
        Since the DummyClassifier doesn't seem to have a verbose option.

        Args:
            X (pd.DataFrame): features
            y (pd.Series): target

        Returns:
            float: [description]
        """
        y_proba = self.do_predict(X)
        return roc_auc_score(y, y_proba)