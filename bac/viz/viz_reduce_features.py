import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Sequence, List
import sys
import logging
import click

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.config import parse_config

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def make_heat_map(
    data: np.ndarray, 
    names: List[str], 
    feature_labels: List[str], 
    figure_fpath: str
    ) -> plt.Figure:
    """Heat map of feature correlations -> Reduce dimensionality

    Args:
        data (np.ndarray): [description]
        names (List): column names
        feature_labels (List): human readable labels for variable names

    Returns:
        plt.Figure: a figure object
    """
    df = pd.DataFrame(data, columns=names)
    corr = df.corr(method='pearson', min_periods=30)
    
    plt.figure(figsize=(24, 24))
    sns.heatmap(data=corr, 
                cmap='twilight', 
                xticklabels=names, yticklabels=names, 
                vmin=-1.0, vmax=1.0)
    sns.set(font_scale=1.4)
    
    fig = plt.gcf()
    plt.savefig(figure_fpath, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()


@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io.yml",
    )
@click.argument(
    "features_config", type=click.Path(exists=True), default="/mnt/configs/features.yml"
)
def main(io_config = "/mnt/configs/io.yml",
         features_config = "/mnt/configs/features.yml"):

    # Load config for file-io
    io_config = parse_config(io_config)
    
    # Load config for feature subsetting
    features_config = parse_config(features_config)
    keep_features = features_config["features_to_keep"][2:]# Exclude label & user_id
    labels_dict = features_config["feature_labels"]
    feature_labels = [labels_dict[feat]["label"] for feat in keep_features]

    # Load Data (only train)
    logging.info(f"\nLoading the partitioned data...")
    train_filename = (io_config["partitions_filenames"])[0]
    dfs_map = load_data_partitions(io_config["partitions_folder"], [train_filename])
    train = dfs_map["train"].iloc[:, 2:]

    # Feature correlation heat map: All Features
    figure_folder = Path(io_config["figures_fpath"])
    if not figure_folder.exists():
        raise ValueError(f"{figure_folder} is not a valid filepath for figures.")
    figure_fpath = str(figure_folder / "corr/feature_correlations_all.pdf")
    
    # 1) Plot Feature Correlations in the entire Feature Set
    # -- assumes first two columns are the label + user_id
    column_names = train.columns
    feature_labels = column_names
    make_heat_map(train, column_names, feature_labels, figure_fpath)
    logging.info(f"\nSaved heatmap of all features to: {figure_fpath}")

    # 2) Plot Feature Correlations in Reduced Feature Set
    figure_outpath = figure_folder / "corr/feature_correlations_reduced.pdf"
    keep_labels = [labels_dict[col]["label"] for col in keep_features]
    keep_train = train[keep_features]
    make_heat_map(keep_train, keep_features, keep_labels, figure_outpath)
    logging.info(f"\nSaved heatmap of reduced feature subset to: {figure_outpath}")        

if __name__ == "__main__":
    main()

