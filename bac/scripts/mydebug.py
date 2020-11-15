from bac.util.io import (load_data)

DATA_FOLDER = '/mnt/data'
FILENAME = 'bac_2019-10-12.parquet'
BACTRACK_FILEPATH = '/'.join([DATA_FOLDER, FILENAME])


df = load_data(BACTRACK_FILEPATH)
print(df.head(6))