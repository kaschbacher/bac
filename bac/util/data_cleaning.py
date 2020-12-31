import pandas as pd
from typing import Sequence
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def split_data(data: pd.DataFrame) -> Sequence[pd.DataFrame]:
    """
    Reformat pd.DataFrame into X_train, y_train, etc
    -- excluding the user_id in column[1].
    """
    X = data.iloc[:, 2:]
    y = data["bac_clinical"]
    y.name = "bac_clinical"
    return X, y


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
        
    logging.info(f"Missing data filled with the value={missing_value}")
    return clean_dfs


def compute_class_imbalance(y: pd.Series) -> None:
    """Compute the class imbalance in a given series of targets (train, test, etc)

    Args:
        y (pd.Series): A series of 0 and 1's for the target
        -- Assumes that cases or targets are labeled 1 and non-cases 0
    """
    vc = y.value_counts()
    non_cases = vc.loc[vc.index==0].values[0]
    cases = vc.loc[vc.index==1].values[0]
    
    logging.info(f"N={cases} cases; N={non_cases} non_cases. Class imbalance non-cases to cases: {non_cases/cases}")