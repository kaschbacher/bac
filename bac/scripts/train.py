# -*- coding: utf-8 -*-
"""
This module is for training models.

Examples:
    Example command line executable::

        $ python train.py "/mnt/configs/io_config.yml"
"""
import logging
import sys
import click
import datetime as dt

from bac.models.lgbm_classifier import LightGBMModel

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.data_cleaning import limit_features, assign_train_dev, fill_missing_data
from bac.util.config import parse_config

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io_config.yml",
)
@click.argument(
    "features_config", type=click.Path(exists=True), default="/mnt/configs/features_config.yml"
)
@click.argument(
    "models_config", type=click.Path(exists=True), default="/mnt/configs/models_config.yml"
)
#@setup_logging_env
def main(
    io_config = "/mnt/configs/io_config.yml", 
    features_config = "/mnt/configs/features_config.yml",
    models_config = "/mnt/configs/models_config.yml"):
    """ Main function that loads config, sets up logging, and trains a model

    Args:
        io_config (str, optional): yaml config for paths to load data
        features_config (str, optional): yaml config for features to exclude
        models_config (str, optional): yaml config for LGBM boosting and training params
    """
    # Set IO Paths
    logger.info(f"Loading config {io_config}")
    config = parse_config(io_config)
    logger.info(f"Config: \n{config}")

    # Load Data
    dfs_map = load_data_partitions(config["partitions_folder"], config["partitions_filenames"][:2])
    X_train, y_train, X_dev, y_dev = assign_train_dev(dfs_map)
    
    # Feature Reduction - to minimize collinearity (reduce_features.py)
    features_to_drop = parse_config(features_config)["features_to_drop"]
    [X_train, X_dev] = limit_features([X_train, X_dev], drop_cols = features_to_drop)
    
    # Fill Missing Data
    [X_train, X_dev] = fill_missing_data([X_train, X_dev], missing_value = -999)
    
    # Load Model Parameters
    models_config = parse_config(models_config)
    boosting_params = models_config["boosting_params"]
    fit_params = models_config["fit_params"]
    fit_params["eval_set"] = [(X_dev, y_dev)]
    
    # Train Model
    model = LightGBMModel(**boosting_params)
    model.do_fit(X_train, y_train, **fit_params)
    
    # Save Model
    model_name = f'lgbm_model_{dt.date.today()}'
    model.do_save(config['models_folder'], model_name)
    
if __name__ == "__main__":
    main()
