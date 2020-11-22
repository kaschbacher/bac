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

def load_feature_labels(feature_label_fpath: str):
    """Read in feature labels from external json"""
    with open(feature_label_fpath, 'r') as json_f:
        return json.load(json_f)
    
    
# def package_data(partitions: dict, names: List, idx_uid: int=1):
#     """ Partition data for XGboost
#     Assign data - (Note: must dropna first, otherwise missing=-999 throws an error because NaNs also still exist)
#     """
    
#     print ("\nSetting up data-matrices for Gradient Boosted Classification Tree with Outcome: {}...\n".format(names[0]))
#     names = [str(name) for name in names]# convert from byte to string
#     del names[idx_uid]# Omit user_id at idx, 1
#     partitions['names']=names
#     partitions['dtrain'] = xgb.DMatrix(data=partitions['X_train'], label=partitions['y_train'], 
#                                        feature_names=names[1:], missing=MISSING_VALUE)
#     partitions['ddev'] = xgb.DMatrix(data=partitions['X_dev'], label=partitions['y_dev'], 
#                                      feature_names=names[1:], missing=MISSING_VALUE)
#     partitions['dtest'] = xgb.DMatrix(data=partitions['X_test'], label=partitions['y_test'], 
#                                       feature_names=names[1:], missing=MISSING_VALUE)
#     return partitions


# def get_balance_weight(y_train):

#     pos = np.sum(y_train, axis=0, dtype=int)
#     neg = y_train.shape[0]-pos
#     print ('\n{} positive cases and {} negative cases'.format(pos, neg))

#     scale_pos_weight = round(neg/pos, 2)# 2.38 for the BACtrack dataset
#     print ('Scale Weight for balanced classes would be: {}'.format(scale_pos_weight))
#     return scale_pos_weight