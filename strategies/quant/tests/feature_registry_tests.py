import pandas as pd
import pandas_ta as ta
import numpy as np
import pytest
from datetime import datetime, timedelta

from test_data_generator import sample_ohlcv_data, feature_params
from strategies.quant.signal_pipeline.feature_registry import FeatureRegistry, FEATURE_MAP


# Define all features to test with: (feature_map_key, function_name, required_parameters)
FEATURES_TO_TEST = [
    ("RSI", "compute_rsi", ["rsi_length"]),
    ("MA_SPREAD", "compute_ma_spread", ["ma_length_short", "ma_length_long"]),
    ("LOG_RETURNS", "compute_log_returns", []),
    ("ZSCORE_DISTANCE", "compute_zscore_distance_rolling_mean", ["zscore_rolling_length"]),
    ("ROLLING_VOLATILITY", "compute_rolling_volatility", ["volatility_length"]),
]


class TestFeatureRegistry:
    """
    Parametrized test suite for feature registry.
    
    For each feature function, we test:
    1. Existence of the function
    2. Existence of the map entry for the function
    3. Correctness of the map (required_parameters, func reference)
    4. Correctness of the output (type, shape)
    """
    
    @pytest.mark.parametrize("feature_key,func_name,required_params", FEATURES_TO_TEST)
    def test_feature_function_exists(self, feature_key, func_name, required_params):
        """
        Test 1: Verify feature function exists in FeatureRegistry.
        """
        assert hasattr(FeatureRegistry, func_name)
        assert callable(getattr(FeatureRegistry, func_name))
    
    @pytest.mark.parametrize("feature_key,func_name,required_params", FEATURES_TO_TEST)
    def test_feature_map_exists(self, feature_key, func_name, required_params):
        """
        Test 2: Verify feature entry exists in FEATURE_MAP.
        """
        assert feature_key in FEATURE_MAP
    
    @pytest.mark.parametrize("feature_key,func_name,required_params", FEATURES_TO_TEST)
    def test_feature_map_correctness(self, feature_key, func_name, required_params):
        """
        Test 3: Verify feature map has correct structure and references.
        """
        entry = FEATURE_MAP[feature_key]
        
        # Check map structure
        assert "func" in entry
        assert "required_parameters" in entry
        
        # Check function reference
        assert callable(entry["func"])
        assert entry["func"] == getattr(FeatureRegistry, func_name)
        
        # Check required parameters
        assert isinstance(entry["required_parameters"], list)
        for param in required_params:
            assert param in entry["required_parameters"]
    
    @pytest.mark.parametrize("feature_key,func_name,required_params", FEATURES_TO_TEST)
    def test_feature_output_correctness(self, feature_key, func_name, required_params, sample_ohlcv_data, feature_params):
        """
        Test 4: Verify feature output is correct (type, shape, value range).
        """
        func = FEATURE_MAP[feature_key]["func"]
        output = func(sample_ohlcv_data, feature_params)
        
        # Check output type and shape
        assert isinstance(output, pd.Series)
        assert len(output) == len(sample_ohlcv_data)
        assert output.index.equals(sample_ohlcv_data.index)
    

    # ========================
    # Edge Case Tests
    # ========================
    # TODO: Test when input dataframe is empty
    # TODO: Test when required columns are missing in the input dataframe
    # TODO: Test when NaN values are present within the input dataframe