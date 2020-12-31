# -*- coding: utf-8 -*-
"""
This module is for training multiple models for model comparison.

Examples:
    Example command line executable::

        $ python train.py "/mnt/configs/io_config.yml"
"""
import logging
import sys
import click
import datetime as dt
import pandas as pd
import numpy as np

from bac.models.lgbm_classifier import LightGBMModel
from bac.models.dummy_classifier import DummyModel
from bac.models.model_schemas import ModelSchemas

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.data_cleaning import limit_features, split_data, fill_missing_data
from bac.util.config import parse_config

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Example CLI usage:
# python bac/scripts/train.py features_config="/mnt/configs/features.yml" --feature_subset=True
    

@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io.yml",
)
@click.argument(
    "features_config", type=click.Path(exists=True), default="/mnt/configs/features.yml"
)
@click.argument(
    "models_config", type=click.Path(exists=True), default="/mnt/configs/train.yml"
)
@click.option("--feature_subset", type=bool, default=True, help="Use a subset of the features given in the features.yml")
#@setup_logging_env
def main(
    io_config = "/mnt/configs/io.yml", 
    features_config = "/mnt/configs/features.yml",
    models_config = "/mnt/configs/train.yml",
    feature_subset = True
    ):
    """ Main function that loads config, sets up logging, and trains a model

    Args:
        io_config (str, optional): yaml config for paths to load data
        features_config (str, optional): yaml config for features to exclude
        models_config (str, optional): yaml config for LGBM boosting and training params
        feature_subset: load only the reduced set of features_to_keep in features.yml
    """
    # Load IO Paths
    logger.info(f"Loading config {io_config}")
    io_cfg = parse_config(io_config)
    logger.info(f"Config: \n{io_cfg}")
    
    # Load Feature & Model Configs
    features_cfg = parse_config(features_config)
    models_cfg = parse_config(models_config)
    
    # Set Model Parameters
    boosting_params = models_cfg["boosting_params"]
    model_folder = io_cfg['models_folder']

    # Load Data
    columns = None
    if feature_subset:
        columns = features_cfg["features_to_keep"]
    datasets = io_cfg["partitions_filenames"][:2]# train & dev only
    dfs_map = load_data_partitions(io_cfg["partitions_folder"], datasets, columns)
    X_train, y_train = split_data(dfs_map['train']) 
    X_dev, y_dev = split_data(dfs_map['dev'])
    # TODO: Implement get n-unique users from dfs_map partitions   
    # -- Add model class method to associate n_users with the model obj 
        
    # Fill Missing Data
    [X_train, X_dev] = fill_missing_data([X_train, X_dev], missing_value = -999)
    
    
    # Train Multiple Models, defined by ModelSchemas Class
    ms = ModelSchemas(X_train.columns, features_cfg)
    
    for model_schema in ms.schemas:
        logging.info(f"\nBuilding model {model_schema['name']}...")
        
        # Output Filename
        model_fname = f"lgbm_model_{model_schema['name']}"
        
        # Set X and y
        subset_columns = model_schema['features']
        target = model_schema["target"]
        X = X_train[subset_columns]
        y = y_train
        
        # Train & Save Models
        fit_params = models_cfg["fit_params"]
        fit_params["eval_set"] = [(X, y), (X_dev[subset_columns], y_dev)]
        fit_params["eval_names"] = ['train', 'dev']
        
        if target == y_train.name:
            clf = LightGBMModel(**boosting_params)
        elif target == 'majority_class':
            dummy_params = {"strategy": "most_frequent"}# "stratified"
            clf = DummyModel(**dummy_params)
        else:
            raise ValueError(f"{target} specified in ModelSchemas \
                does not match y_train: {y_train.name}")
        
        clf.do_fit(X, y, **fit_params)
        clf.save_training_info(X)
        # DummyClassifier doesn't automatically log the aucs
        if target == 'majority_class':
            logging.info(f'Train auc: {clf.get_auc(X, y)}.') 
            logging.info(f'Eval auc: {clf.get_auc(X_dev[subset_columns], y_dev)}')
        
        clf.do_save(model_folder, model_fname)
    
if __name__ == "__main__":
    main()
