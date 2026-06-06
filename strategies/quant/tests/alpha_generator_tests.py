import pandas as pd
import pandas_ta as ta
import numpy as np
import pytest
from datetime import datetime, timedelta

from test_data_generator import sample_ohlcv_data, feature_params
from strategies.quant.signal_pipeline.feature_registry import FEATURE_MAP
import strategies.quant.signal_pipeline.alpha_generator as alpha_generator
from strategies.quant.signal_pipeline.alpha_generator import STRATEGY_MAP


# Define all strategies to test with: (strategy_map_key, function_name, required_features)
STRATEGIES_TO_TEST = [
    ("TREND_FOLLOWING", "trend_following_alpha", ["RSI"]),
    ("MEAN_REVERSION", "rsi_mean_reversion_alpha", ["RSI"]),
]


class TestAlphaGenerator:
    """
    Parametrized test suite for alpha generator.
    
    For each strategy function, we test:
    1. Existence of the function
    2. Existence of the map entry for the function
    3. Correctness of the map (required_features, func reference)
    4. Correctness of the output (type, shape, value range)
    """

    @pytest.mark.parametrize("strategy_key,func_name,required_features", STRATEGIES_TO_TEST)
    def test_strategy_function_exists(self, strategy_key, func_name, required_features):
        """
        Test 1: Verify strategy function exists.
        """
        assert hasattr(alpha_generator, func_name)
        assert callable(getattr(alpha_generator, func_name))

    @pytest.mark.parametrize("strategy_key,func_name,required_features", STRATEGIES_TO_TEST)
    def test_strategy_map_exists(self, strategy_key, func_name, required_features):
        """
        Test 2: Verify strategy entry exists in STRATEGY_MAP.
        """
        assert strategy_key in STRATEGY_MAP
    
    @pytest.mark.parametrize("strategy_key,func_name,required_features", STRATEGIES_TO_TEST)
    def test_strategy_map_correctness(self, strategy_key, func_name, required_features):
        """
        Test 3: Verify strategy map has correct structure and references.
        """
        entry = STRATEGY_MAP[strategy_key]
        
        # Check map structure
        assert "alpha_func" in entry
        assert "required_features" in entry
        
        # Check function reference
        assert callable(entry["alpha_func"])
        assert entry["alpha_func"] == getattr(alpha_generator, func_name)
        
        # Check required features
        assert isinstance(entry["required_features"], list)
        for feature in required_features:
            assert feature in entry["required_features"]

    @pytest.mark.parametrize("strategy_key,func_name,required_features", STRATEGIES_TO_TEST)
    def test_strategy_output_correctness(self, strategy_key, func_name, required_features, sample_ohlcv_data, feature_params):
        """
        Test 4: Verify the correctness of the output (type, shape, value range)
        """
        func = STRATEGY_MAP[strategy_key]["alpha_func"]
        feature_matrix = sample_ohlcv_data.copy()
        for feature in STRATEGY_MAP[strategy_key]["required_features"]:
            feature_matrix[feature] = FEATURE_MAP[feature]["func"](sample_ohlcv_data, feature_params)
        
        out = func(feature_matrix)

        assert isinstance(out, pd.DataFrame)
        assert "alpha_score" in out.columns
        assert len(out) == len(feature_matrix)
        # TODO: Verify alpha_score values are within expected range [-1.0, 1.0]
        assert (out["alpha_score"] >= -1.0).all() and (out["alpha_score"] <= 1.0).all()