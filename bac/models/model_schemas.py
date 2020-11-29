from typing import Sequence

class ModelSchemas():
    """A class to define the models we want to compare,
    based on different feature sets and/or targets.
    Intended to be used by train_multiple.py
    """
    
    def __init__(self, X_columns: Sequence[str], features_config: dict):
        """Builds "schemas" a list of dictionaries

        Args:
            X_columns (Sequence[str]): the columns of X_train
            feature_config (dict): config parsed from features.yml
        """
        self.features = X_columns
        self.config = features_config
        self._define_schemas()
        

    def _define_schemas(self):
        """Define self.schemas as a List of dictionaries
        Each dictionary defines a model in a set to be compared. 
        Remember that features should not contain the user_id.

        Args:
            self (ModelSelectionSchemas): object holds base feature set, features.yml, schemas
        """
        schemas = []
        
        model = {}
        model['name'] = 'all_w_majority_class'
        model['features'] = self.features
        model['target'] = 'majority_class'
        schemas.append(model)
        
        model = {}
        model['name'] = 'all'
        model['features'] = self.features
        model['target'] = 'bac_clinical'
        schemas.append(model)
        
        model = {}
        model['name'] = 'all_but_estimate'
        bac_estimate = self.config["bac_estimate"]
        model['features'] = self.features.difference(set(bac_estimate))
        model['target'] = 'bac_clinical'
        schemas.append(model)
        
        model = {}
        model['name'] = 'all_but_bacs'
        bac_based_features = self.config["bac_measures"]
        model['features'] = self.features.difference(set(bac_based_features))
        model['target'] = 'bac_clinical'
        schemas.append(model)
        
        model = {}
        model['name'] = 'bac_estimate'
        model['features'] = bac_estimate
        model['target'] = 'bac_clinical'
        schemas.append(model)
        
        self.schemas = schemas
        