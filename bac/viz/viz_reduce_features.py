import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Sequence, List
import sys
import logging
import click

from bac.util.io import load_data_partitions, load_feature_labels
from bac.util.config import parse_config

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def make_heat_map(data: np.ndarray, names: List[str], feature_labels: List[str]) -> plt.Figure:
    """Heat map of feature correlations -> Reduce dimensionality

    Args:
        data (np.ndarray): [description]
        names (List): column names
        feature_labels (List): human readable labels

    Returns:
        plt.Figure: a figure object
    """
    df = pd.DataFrame(data, columns=names)
    corr = df.corr(method='pearson', min_periods=30)
    
    plt.figure(figsize=(24,24))
    sns.heatmap(data=corr, 
                cmap='twilight', 
                xticklabels=names, yticklabels=names, 
                vmin=-1.0, vmax=1.0)
    sns.set(font_scale=1.4)
    
    fig = plt.gcf()
    return fig

@click.command()
@click.argument(
    "io_config", type=click.Path(exists=True), default="/mnt/configs/io.yml",
    )
@click.argument(
    "features_config", type=click.Path(exists=True), default="/mnt/configs/drop_features.yml"
)
def main(io_config = "/mnt/configs/io.yml",
         features_config = "/mnt/configs/drop_features.yml"):

    # Load config for file-io
    config = parse_config(io_config)
    feature_labels = load_feature_labels(config["feature_labels_fpath"])
    
    # Load config for feature subsetting
    features_config = parse_config(features_config)
    features_to_drop = features_config["features_to_drop"]

    # Load Data (only train)
    logging.info(f"\nLoading the partitioned data...")
    train_filename = (config["partitions_filenames"])[0]
    dfs_map = load_data_partitions(config["partitions_folder"], [train_filename])

    # Feature correlation heat map: All Features
    train = dfs_map["train"]
    names = train.columns[2:]# assumes first two columns are not features
    fig = make_heat_map(train, names, feature_labels)

    fig_folder = config["figures_fpath"]
    figpath = os.path.join(fig_folder, "corr/feature_correlations_all.pdf")
    plt.savefig(figpath, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved heatmap of all features to: {figpath}")

    # Reduce the Feature Set & Resave
    keep_names = [name for name in names if name not in features_to_drop]

    # Replot Feature Correlations in Reduced Feature Set
    fig = make_heat_map(train, keep_names, feature_labels)
    figpath = os.path.join(fig_folder, "corr/feature_correlations_reduced.pdf")
    plt.savefig(figpath, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved heatmap of reduced feature subset to: {figpath}\n")        

if __name__ == "__main__":
    main()

