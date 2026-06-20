import pandas as pd

from strategies.quant.configs.active_config import ACTIVE_CONFIG
from strategies.quant.signal_pipeline.execute import execute


# Perform trades with the specified input data, metadata and active config
def trade(
        input_data_path: str, 
        input_metadata_path: str
    ) -> dict:
    input_data = pd.read_csv(input_data_path)
    input_metadata = pd.read_csv(input_metadata_path)

    return execute(input_data, input_metadata, ACTIVE_CONFIG)