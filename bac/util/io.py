import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the main bac dataset, in parquet format

    Args:
        filepath (str): local path to data

    Returns:
        pd.DataFrame: dataframe formatted data
    """
    df = pd.read_parquet(filepath, engine="pyarrow")
    print (df.shape, '\n')# QA
    
    # QA datafile organization
    assert (df.shape[0]>0) & ~(df.empty)
    assert df.columns[0]=='bac_clinical'
    assert df.columns[1]=='user_id'
    
    return df