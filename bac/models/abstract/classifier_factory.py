from typing import Any
import pandas as pd

from bac.models.classifier_base import ClassifierBase
from bac.models.sklearn_classifier import SklearnClassifier


class ClassifierFactory:
    
    known_models = {
        "sklearn_classifier": SklearnClassifier
    }
    
    def build_model_from_name(
        self,
        model_name: str,
        model_params: Dict[str, Any],
    ) -> ClassifierBase:
        """Creates a model instance.
        Key is the name, provides params.

        Args:
            model_name: name of a classifier model ('known_models')
            model_params: dictionary of keyword arguments for model
        """
        model_class = self.known_models.get(model_name)
        if not model_class:
            raise ValueErrors(f"Model {model_name} not Implemented.")
        return model_class(**model_params)