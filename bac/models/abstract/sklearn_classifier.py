from typing import Any

import pandas as pd

from bac.models.classifier_base import ClassifierBase
from bac.models.sklearn_classifier import SklearnClassifier


class ClassifierFactory:
    
    known_models = {
        'lgbc': LGBMCClassifier
    }
    
    def __init__(self, classifier_name: str, **kwargs):
        """Create a model. Pass kwargs to sklearn model

        Args:
            classifier_name: one of the known models.
            -- has fit() and predict_proba() methods
            **kwargs: keyword args for classifier
        """
        super().__init__()
        model_class = self.known_models.get(classifier_name)
        if model_class is None:
            raise ValueError(f"Unknown model {classifier_name}")
        
    def do_fit(self, x: pd.DataFrame, y)