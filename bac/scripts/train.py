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
    "feature_config", type=click.Path(exists=True), default="/mnt/configs/features.yml"
)
@click.argument(
    "models_config", type=click.Path(exists=True), default="/mnt/configs/train.yml"
)
@click.option("--feature_subset", type=bool, default=True, help="Use a subset of the features given in the features.yml")
#@setup_logging_env
def main(
    io_config = "/mnt/configs/io.yml", 
    feature_config = "/mnt/configs/features.yml",
    models_config = "/mnt/configs/train.yml",
    feature_subset = True):
    """ Main function that loads config, sets up logging, and trains a model

    Args:
        io_config (str, optional): yaml config for paths to load data
        features_config (str, optional): yaml config for features to exclude
        models_config (str, optional): yaml config for LGBM boosting and training params
        feature_subset: load only the reduced set of features_to_keep in features.yml
    """
    # Set IO Paths
    logger.info(f"Loading config {io_config}")
    config = parse_config(io_config)
    logger.info(f"Config: \n{config}")

    # Load Data
    columns = None
    if feature_subset:
        columns = parse_config(feature_config)["features_to_keep"]
    datasets = config["partitions_filenames"][:2]# train & dev only
    dfs_map = load_data_partitions(config["partitions_folder"], datasets, columns)
    X_train, y_train = split_data(dfs_map['train']) 
    X_dev, y_dev = split_data(dfs_map['dev'])
    # TODO: Implement get n-unique users from dfs_map partitions   
    # -- Add model class method to associate n_users with the model obj 
        
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
    model.save_training_info(X_train)
    
    # Save Model
    model_name = f'lgbm_model_{dt.date.today()}'
    model.do_save(config['models_folder'], model_name)
    
if __name__ == "__main__":
    main()
