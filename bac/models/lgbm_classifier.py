import pandas as pd
from lightgbm import LGBMClassifier
import joblib
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class LightGBMModel():
    
    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)
        self.fitted = False
        self.columns = []
        self.boosting_params = kwargs
        self.fit_args = {}
        self.best_n_estimators = None
        

    def do_fit(self, X_train: pd.DataFrame, y_train: pd.Series, **fit_args) -> None:
        """Train model. Extends model_base abstract class

        Args:
            X_train (pd.DataFrame): feature set (user-day)
            y_train (pd.Series): targets. required.
            **fit_args:  kwargs for LGBM.fit() method
            -- e.g., eval_set, eval_metric
        """
        logging.info("\nTraining LGBM...")
        logger.info(f"Boosting Params: \n{self.boosting_params}")
        #logger.info(f"LGBM Fit Params: \n{fit_args}\n")
        
        self.model.fit(X_train.values, y_train.values.ravel(), **fit_args)
        self.fitted = True
        self.fit_args = fit_args

    
    def save_training_info(self, X_train: pd.DataFrame):
        """Save columns used in training as instance variables

        Args:
            X_train (pd.DataFrame): feature set for training
        """
        assert self.fitted
        self.columns = X_train.columns.tolist()
        self.n_userdays = len(X_train)
        self.best_n_estimators = self.model.best_iteration_
        logging.info(f"\nBest Model Iteration: {self.best_n_estimators}\n")
    
    
    def do_predict(self, X_eval: pd.DataFrame) -> pd.Series:
        """ 
        Return probability scores [0, 1] for each row of X.
        First checks that columns match self.columns
        """
        scores = self.model.predict_proba(X_eval.values)
        if len(scores.shape) == 2:
            scores = scores[:, 1]
        return pd.Series(scores, index=X_eval.index)
    
    
    def do_save(self, model_folder: str, model_name: str):
        """Save a serialized model to a given folder

        Args:
            model_folder: location on the docker volume to save
            model_name: name of the serialized model file
        """
        model_outpath = f"{model_folder}/{model_name}.joblib"
        joblib.dump(self, model_outpath)
        logger.info(f"Completed. Model saved to docker: {model_outpath}\n")

    