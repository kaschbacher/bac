import pandas as pd
import os
from typing import Sequence, List
import json
import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_data(filepath: str) -> pd.DataFrame:
    """Load the main bac dataset, in parquet format

    Args:
        filepath (str): local path to data

    Returns:
        pd.DataFrame: dataframe formatted data
    """
    df = pd.read_parquet(filepath, engine="pyarrow")
    logging.info(df.shape, '\n')# QA
    
    # QA datafile organization
    assert (df.shape[0]>0) & ~(df.empty)
    assert df.columns[0]=='bac_clinical'
    assert df.columns[1]=='user_id'
    
    return df

def load_data_partitions(folder: str, filenames: Sequence[str])->dict:
    """Load the train, dev, test parquet files from local"""
    dfs_map = {}
    if os.path.exists(folder):
        for filename in filenames:
            filepath = os.path.join(folder, filename)
            logging.info(f"Loading {filepath}...")
            df = pd.read_parquet(filepath, engine="pyarrow")
            name = filename.replace(".parquet", "")
            dfs_map[name] = df
            logging.info(f"DF {name} loaded with shape: {df.shape}")
    return dfs_map

def clean_data_partitions(dfs_map: dict) -> dict:
    """No missing target data allowed."""
    dfs_map_clean = {}
    for key, df in dfs_map.items():
        df = df.dropna(subset=df.columns[0])
        dfs_map_clean[key] = df
    return dfs_map_clean

def assign_train_dev(dfs_map: dict) -> Sequence[pd.DataFrame]:
    """
    Reformat dictionary into X_train, y_train, etc
    -- excluding the user_id in column[1].
    """
    train = dfs_map["train"]
    dev = dfs_map["dev"]
    X_train = train.iloc[:, 2:]
    y_train = train["bac_clinical"]
    X_dev = dev.iloc[:, 2:]
    y_dev = dev["bac_clinical"]
    return X_train, y_train, X_dev, y_dev

def fill_missing_data(
    dfs_list: Sequence[pd.DataFrame], columns: Sequence[str] = None, missing_value: int = -999
    ) -> Sequence[pd.DataFrame]:
    """Fill in missing data with a specified value.

    Args:
        features (Sequence[dfs]): a list of multiple dfs - train, dev, etc
        columns: a list of column names to fill
        missing_value (int): value to fill

    Returns:
        Sequence[dfs]: [description]
    """
    assert isinstance(missing_value, int)
    clean_dfs = []
    for df in dfs_list:
        if columns is None:
            columns = df.columns
        df[columns] = df[columns].fillna(value = missing_value)
        clean_dfs.append(df)
    return clean_dfs

def load_feature_labels(feature_label_fpath: str):
    """Read in feature labels from external json"""
    with open(feature_label_fpath, 'r') as json_f:
        return json.load(json_f)
    
def limit_features(
    dfs: Sequence[pd.DataFrame], 
    drop_cols: Sequence[str]=None, 
    keep_cols: Sequence[str]=None
    ) -> Sequence[pd.DataFrame]:
    """ 
    Limits the feature set of multiple df partitions of the dataset.
    Assumes the input dfs are X_train, X_dev, X_test with no user id or y
    
    Allows either dropping columns or keeping columns
    """
    out_dfs = []
    for df in dfs:
        if drop_cols is not None:
            remaining_cols = set(df.columns).difference(drop_cols)
            df = df[remaining_cols]
        if keep_cols is not None:
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]
        out_dfs.append(df)
    return out_dfs
