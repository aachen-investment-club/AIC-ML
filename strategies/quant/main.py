import pandas as pd
import numpy as np
from config import ACTIVE_CONFIG, PORTFOLIO_NAME, SHARES_TO_TRADE
from strategies.quant.signal_pipeline.alpha_generator import STRATEGY_MAP
from strategies.quant.signal_pipeline.feature_registry import FEATURE_MAP
from strategies.quant.signal_pipeline.signal_generator import generate_signals, CONTEXT_FEATURE_MAP
from strategies.quant.utils.interface import input_adapter, output_adapter


def main(data_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict:
    # Adapt input data
    matrix = input_adapter(data_df)

    # Alpha Feature assembly
    strategy = ACTIVE_CONFIG["active_strategy"]
    alpha_func = STRATEGY_MAP[strategy]["alpha_func"]
    
    required_features = STRATEGY_MAP[strategy]["required_features"]
    for feature_key in required_features:
        func = FEATURE_MAP[feature_key]["func"]
        extracted_params = {
            param_name: ACTIVE_CONFIG["alpha_parameters"][param_name]
                for param_name in FEATURE_MAP[feature_key]["required_parameters"]
        }

        feature_series = func(matrix, extracted_params)
        feature_series.name = feature_key
        matrix[feature_key] = feature_series
        
    # Alpha score generation
    alpha_series = alpha_func(matrix)
    matrix["alpha_score"] = alpha_series

    # Context Feature assembly
    for context_feature_key in CONTEXT_FEATURE_MAP:
        func = CONTEXT_FEATURE_MAP[context_feature_key]["func"]
        extracted_params = {
            param_name: ACTIVE_CONFIG["context_parameters"][param_name]
                for param_name in CONTEXT_FEATURE_MAP[context_feature_key]["required_parameters"]
        }

        context_feature_series = func(matrix, extracted_params)
        context_feature_series.name = context_feature_key
        matrix[context_feature_key] = context_feature_series

    # Signal generation
    signal_df = generate_signals(matrix, ACTIVE_CONFIG["context_parameters"])

    # Adapt output signals
    return output_adapter(signal_df, meta_df, PORTFOLIO_NAME, SHARES_TO_TRADE)