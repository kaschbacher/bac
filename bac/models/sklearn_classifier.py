import pandas as pd
from lightgbm import LGBMClassifier

from bac.models.classifier_base import ClassifierBase
#from bac.util.sklearn_utils import compute_probabilities
# Implementing below in do_predict() instead


class SKlearnClassifier(ClassifierBase):
    
    known_models = {
        "lgbm": LGBMClassifier
    }
    
def __init__(self, classifier_name: str="lgbm", **kwargs):
    """Instantiate a model. Pass kwargs to underlying model.

    Args:
        classifier_name: One of the known_models. Defaults to "lgbm".
        -- has fit() and predict_proba() methods
        **kwargs: keyword arguments passed to the sklearn classifier
    """
    super().__init__()
    model_class = self.known_models.get(classifier_name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {classifier_name}")
    self.model = model_class(**kwargs)
    
def do_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Train model. Extends model_base abstract class

    Args:
        X_train (pd.DataFrame): feature set (user-day)
        y (pd.Series): targets. required.
    """
    self.model.fit(X_train.values, y_train.values.ravel())
    
def do_predict(self, X_eval: pd.DataFrame) -> pd.Series:
    """ 
    Return probability scores [0, 1] for each row of X.
    First checks that columns match self.columns
    """
    scores = self.model.predict_proba(X_eval.values)
    if len(scores.shape) == 2:
        scores = scores[:, 1]
    return pd.Series(scores, index=X_eval.index)