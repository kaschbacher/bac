import pandas as pd
import os
from typing import Sequence, List, Union
import json
import yaml
import sys
import logging
import joblib
from pathlib import Path


from bac.models.lgbm_classifier import LightGBMModel
from bac.models.dummy_classifier import DummyModel

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_data(filepath: str, columns: Sequence[str]=None) -> pd.DataFrame:
    """Load the main bac dataset, in parquet format

    Args:
        filepath (str): local path to data
        columns:  sequence of str col-names to load, or None to load all

    Returns:
        pd.DataFrame: dataframe formatted data
    """
    df = pd.read_parquet(filepath, columns=columns, engine="pyarrow")
    logging.info(df.shape, '\n')# QA
    
    # QA datafile organization
    assert (df.shape[0]>0) & ~(df.empty)
    assert df.columns[0]=='bac_clinical'
    assert df.columns[1]=='user_id'
    
    return df


def load_data_partitions(
    folder: str, 
    filenames: Sequence[str], 
    columns: Sequence[str] = None
    ) -> dict:
    """Load the train, dev, test parquet files from local

    Args:
        folder (str): Docker volume folder with train, dev, test parquets
        filenames (Sequence[str]): Allows loading of one or some of the parquets
        -- names are specified in io_config: e.g., 'train.parquet'
        columns: Subset of columns to load. If None, load all

    Returns:
        dict: maps the name ("train.parquet") onto a pd.DataFrame
    """
    dfs_map = {}
    if os.path.exists(folder):
        for filename in filenames:
            filepath = os.path.join(folder, filename)
            logging.info(f"Loading {filepath}...")
            df = pd.read_parquet(filepath, columns=columns, engine="pyarrow")
            name = filename.replace(".parquet", "")
            dfs_map[name] = df
            logging.info(f"DF {name} loaded with shape: {df.shape}")
    return dfs_map


def load_feature_labels(feature_label_fpath: str) -> dict:
    """Read in feature labels from external json"""
    if Path(feature_label_fpath).exists():
        with open(feature_label_fpath, 'r') as json_f:
            feature_map = json.load(json_f)
            return feature_map
    else:
        logging.info(f"{feature_label_fpath} is not a valid path.")
        return None
    
    
def write_feature_labels_as_yaml(feature_map: dict, feature_label_fpath: str):
    """Helper function to take the feature_labels.json (read above),
    and write it back out as a yaml, to be consistent with codebase.

    Args:
        feature_map (dict): a mapper from feature variable name to label
        feature_label_fpath (str): a filepath to the feature_labels.json
    """
    feature_folder = Path(feature_label_fpath).parent
    if feature_folder.exists():
        new_feature_label_fpath = str(feature_folder / 'feature_labels.yml')
        yml_file = open(new_feature_label_fpath, 'w+')
        yaml.dump(feature_map, yml_file)
        logging.info(f"Finished writing yaml to: {new_feature_label_fpath}.")
    else:
        logging.info(f"{feature_folder} is not a valid path.")
    
    
def load_model(model_fpath: str) -> Union[LightGBMModel, DummyModel]:
    """Load a saved serialized model object

    Args:
        model_fpath (str): local filepath to model object in docker volume
        -- '/volume/data/models/lgbm_model_{model_name}.joblib'

    Returns:
        Union[LightGBMModel, DummyModel]: returns a model obj
    """
    model_fpath = Path(model_fpath)
    if model_fpath.exists():
        logging.info(f"Loading model obj: {model_fpath}...")
        model = joblib.load(model_fpath)
        return model
    else:
        raise ValueError(f"Filepath {model_fpath} does not exist.")