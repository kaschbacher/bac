from typing import Sequence
import pandas as pd
import numpy as np

import logging
import sys
import click
import joblib
from pathlib import Path

from bac.util.config import parse_config
from bac.util.io import load_data_partitions, load_feature_labels, load_model
from bac.util.data_cleaning import split_data, fill_missing_data
from bac.features.feature_importances import compute_feature_importances

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io.yml",
)
@click.argument(
    "features_config", type=click.Path(exists=True), default="/mnt/configs/features.yml"
)
@click.argument(
    "eval_config", type=click.Path(exists=True), default='/mnt/configs/evaluate.yml'
)
def main(
    io_config = "/mnt/configs/io.yml", 
    features_config = "/mnt/configs/features.yml",
    eval_config = '/mnt/configs/evaluate.yml'
    ):
    """Load serialized models and evaluate

    Args:
        io_config: filepaths to load and save data
        features_config: Used to define feature sets for each model
        eval_config: filepath base for saved serialized models
    """
    
    # Load Configs:  Paths, Features, Saved Model-filepaths
    # TODO: eval_config can be part of io_config, if it doesn't need more args
    logger.info(f"Loading configs...")
    io_cfg = parse_config(io_config)
    features_cfg = parse_config(features_config)
    eval_cfg = parse_config(eval_config)
    models_base_fpath = eval_cfg["models_fpath"]
    figures_fpath = io_cfg["figures_fpath"]
    feature_labels = features_cfg["feature_labels"]
    
    # Load Features & Targets
    columns = features_cfg["features_to_keep"]
    test_set = io_cfg["partitions_filenames"][2]
    dfs_map = load_data_partitions(io_cfg["partitions_folder"], [test_set], columns)
    X_test, y_test = split_data(dfs_map['test']) 
    [X_test] = fill_missing_data([X_test], missing_value = -999)

	# Load final model
    model_name = "all"
    model_fpath = models_base_fpath.format(model_name=model_name)
    model = load_model(model_fpath)
    
    # Get predicted probabilities
    y_prob = model.do_predict(X_test)
                
    # Plot Shap Values for Best Model
    base_folder = Path(io_cfg["figures_fpath"])
    output_folder = base_folder / 'shap/'
    shap_df = compute_feature_importances(model.model, X_test, feature_labels, output_folder)
    
    logging.info(f"\nComplete.")

if __name__=="__main__":
    main()