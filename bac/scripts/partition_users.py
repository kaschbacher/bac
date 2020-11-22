import pandas as pd
import numpy as np
import os
from typing import Sequence, List

from bac.util.io import (load_data)


# Define train-val-test split
TRAIN_P, DEV_P = .70, .10
SPLIT = [TRAIN_P, DEV_P, 1 - (TRAIN_P + DEV_P)]

SEED = 3
# BACTRACK_FILEPATH = '/mnt/data/bac_2019-10-12.parquet'
# OUTPUT_FOLDER = '/mnt/data'
BACTRACK_FILEPATH = '/mnt/data/bac_2019-10-12.parquet'
OUTPUT_FOLDER = '/volume/data'


def shuffle_uids(uuids: np.ndarray) -> Sequence[int]:
    """Shuffle unique users for partitioning to train-val-test

    Args:
        uuids (np.array): unique users
        
    Returns:
        np.ndarray: shape = (n-unique users, )
    """
    print (f'Number of unique users: {uuids.shape[0]}')
    np.random.seed(seed=SEED)
    np.random.shuffle(uuids)
    return uuids


def qa_user_partitions(train: np.ndarray, dev: np.ndarray, test: np.ndarray): 
    """QA user-randomization. Ensure no users are in more than one array"""
    # Could condense, but this is somewhat easier to read
    if np.intersect1d(dev, train, assume_unique=True).size>0:  
        print ('Problem with randomization: users in dev and train overlap')
    elif np.intersect1d(dev, test, assume_unique=True).size>0:
        print ('Problem with randomization: users in dev and test overlap')
    elif np.intersect1d(test, train, assume_unique=True).size>0:
        print ('Problem with randomization: users in test and train overlap')
        print (set(train).intersection(set(test)))
    else:
        print ('\nUser Randomization partitions passed the QA tests; Each group contains unique users\n')


def partition_users(uuids: np.ndarray, split: List) -> dict:
    """Splits an array of unique user-ids into train-dev-test

    Args:
        uuids (np.ndarray): [unique user-ids]
        split (List): [train, dev, test proportions]

    Returns:
        dict: [description]
    """
    train_per, dev_per, test_per = split
    s = len(uuids)
    train_ui, dev_ui, test_ui = np.split(uuids, [int(train_per*s), int((train_per+dev_per)*s)])
    
    assert len(uuids) == train_ui.shape[0] + dev_ui.shape[0] + test_ui.shape[0]
    qa_user_partitions(train_ui, dev_ui, test_ui)
    user_id_partitions = {'train': train_ui, 'dev': dev_ui, 'test': test_ui}
    return user_id_partitions


def main():
    # Load Data
    df = load_data(BACTRACK_FILEPATH)
    names = df.columns.tolist()
    print(df.head(6))
    
    # Randomize & partition unique users
    unique_uids = np.unique(df["user_id"].values)
    unique_uids = shuffle_uids(unique_uids)
    user_id_partitions = partition_users(unique_uids, SPLIT)

    final_shape = df.shape[0]
    
    #Save partiitons as separate parquet files for model training
    for name, user_id_arr in user_id_partitions.items():
        # Build a df for a subset of users in that partition
        user_partition = user_id_partitions[name]
        partition_df = df.query("user_id in @user_partition")
        
        # Save out
        filename = os.path.join(OUTPUT_FOLDER, name+".parquet")
        partition_df.to_parquet(filename, compression="snappy")
        final_shape -= partition_df.shape[0]
        print(f"Saved: {filename} with shape {partition_df.shape} and n-unique-users {partition_df.user_id.nunique()}")
        
    # Last QA
    if final_shape != 0:
        raise ValueError(f"Error: Partition shapes do not sum to the whole.")
        
if __name__=="__main__":
    main()