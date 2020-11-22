# -*- coding: utf-8 -*-
"""
This module is for training models without requireing a base class.

Examples:
    Example command line executable::

        $ python train.py config.yml
"""
import logging
import sys
import click
import datetime as dt

from lightgbm import LGBMClassifier
import joblib

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.config import parse_config

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True), default="/mnt/configs/io_config.yml"
)
#@setup_logging_env
def main(config_file="/mnt/configs/io_config.yml"):
    """
    Main function that loads config, sets up logging, and trains a model

    Args:
        config_file (str): path to config file (for logging)

    Returns:
        None
    """
    # Set Parameters
    logger.info(f"Loading config {config_file}")
    config = parse_config(config_file)
    logger.info(f"Config: \n{config}")

    # Load Data
    dfs_map = load_data_partitions(config["partitions_folder"], config["partitions_filenames"])
    train = dfs_map["train"]
    dev = dfs_map["dev"]
    
    # Limit Features
    
    # Fill Missing Data
    
    # Model Inputs
    X_train = train.iloc[:, 2:]
    y_train = train["bac_clinical"]
    X_dev = dev.iloc[:, 2:]
    y_dev = dev["bac_clinical"]
    
    logger.info("Training...")
    boosting_params = {
        "silent": False, 
        "class_weight": "balanced",
        "n_estimators": 30
        }
    
    # Train Model
    model = LGBMClassifier(**boosting_params)
    model.fit(X_train, y_train, 
                     eval_set=[(X_dev, y_dev)], 
                     eval_metric="auc",
                     early_stopping_rounds=5)
    print(model.best_iteration_)
    
    # Save
    model_name = f'lgb_model_{dt.date.today()}'
    model_outpath = f"{config['models_folder']}/{model_name}.joblib"
    joblib.dump(model, model_outpath)

    logger.info(f"Completed. Model saved to docker: {model_outpath}")

if __name__ == "__main__":
    main()
