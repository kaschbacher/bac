# -*- coding: utf-8 -*-
"""
This module is for running predictions.

Examples:
    Example command line executable::

        $ python predict.py
"""
import logging
import click
import joblib
import os

from bac.util.config import parse_config
from bac.util.io import load_data_partitions
from bac.util.data_cleaning import limit_features, split_data, fill_missing_data

logger = logging.getLogger(__name__)


# Example CLI usage:
# python bac/scripts/predict.py io_config=/mnt/configs/io.yml feature_config=/mnt/configs/features.yml --feature_subset=True


@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io.yml",
    )
@click.argument(
    "feature_config", type=click.Path(exists=True), default="/mnt/configs/features.yml"
)
@click.option(
    "--feature_subset", type=bool, default=True, 
    help="Use only a subset of the features, given in features.yml"
)
def main(
    io_config="/mnt/configs/io.yml", 
    feature_config = "/mnt/configs/features.yml",
    feature_subset = True):
    """Main function that loads config, sets up logging, and runs predictions

    Args:
        io_config: config for file paths to load or save.
        feature_config: config for feature sets included in various models.
        feature_subset: bool. If true, uses features_to_keep from yml
    """
    # Configure IO Paths
    logger.info(f"Loading config {io_config}")
    config = parse_config(io_config)

    # Load Data for dev set
    data_folder = config["partitions_folder"]
    filenames = config["partitions_filenames"]
    dev_filename = [filenames[1]]
    
    columns = None
    if feature_subset:
        columns = parse_config(feature_config)["features_to_keep"]
    dfs_map = load_data_partitions(data_folder, dev_filename, columns)
    X_eval, y_eval = split_data(dfs_map['dev'])
    
    logger.info(f"X_eval has shape = {X_eval.shape}")
    
    # Fill Missing Data
    [X_eval] = fill_missing_data([X_eval], missing_value = -999)    
    
    # Load Model    
    model = joblib.load(config["model_predict_fpath"])
    #print(model.__dict__.keys())
    
    # Predict
    logger.info("Predicting")
    scores = model.do_predict(X_eval)
    
    # # TODO: Decide on an org system for experiments
    # # if exp folder doesn't exit, mkdir it
    # experiment_name = "testing_predict"
    # output_folder = f"/volume/data/{experiment_name}"
    # scores.to_parquet(f"{output_folder}/scores.parquet")
    

if __name__ == "__main__":
    main()
