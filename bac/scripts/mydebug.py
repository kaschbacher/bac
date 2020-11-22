from bac.util.io import (load_data)
from bac.util.config import parse_config

# DATA_FOLDER = '/mnt/data'
# FILENAME = 'bac_2019-10-12.parquet'
# BACTRACK_FILEPATH = '/'.join([DATA_FOLDER, FILENAME])


# df = load_data(BACTRACK_FILEPATH)
# print(df.head(6))

io_config = "/mnt/configs/io_config.yml"
config = parse_config(io_config)
print(config["feature_label_fpath"])