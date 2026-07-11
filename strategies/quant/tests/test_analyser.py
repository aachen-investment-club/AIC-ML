import pytest
import pandas as pd
import numpy as np
from strategies.quant.optimize import analyser as analyser_module
from strategies.quant.optimize.analyser import analyse
from strategies.quant.tests.test_data_generator import *



# Define all analysis functions to test with: (analysis_key, function_name, required_arguments)
ANALYSIS_TO_TEST = [
    ("SHARPE_RATIO", "analyse", ["ticker_data", "trade_logs", "risk_free_rate"]),
]


class TestPortfolioAnalyser:
    """
    Parametrized test suite for portfolio analyser.
    
    For each analysis function, we test:
    1. Existence of the function
    2. Existence of the function within the module namespace
    3. Correctness of the function arguments
    4. Correctness of the output (type, value range)
    """

    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_function_exists(self, analysis_key, func_name, required_args):
        """
        Test 1: Verify analysis function exists.
        """
        assert hasattr(analyser_module, func_name)
        assert callable(getattr(analyser_module, func_name))

    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_map_exists(self, analysis_key, func_name, required_args):
        """
        Test 2: Verify analysis entry is active in module namespace.
        """
        assert func_name in dir(analyser_module)
    
    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_arguments_correctness(self, analysis_key, func_name, required_args):
        """
        Test 3: Verify analysis function accepts all required arguments.
        """
        import inspect
        func = getattr(analyser_module, func_name)
        sig = inspect.signature(func)
        
        for arg in required_args:
            assert arg in sig.parameters

    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_output_correctness(self, analysis_key, func_name, required_args, sample_ticker_data, sample_trade_logs):
        """
        Test 4: Verify the correctness of the output (type, value range)
        """
        func = getattr(analyser_module, func_name)
        
        out = func(sample_ticker_data, sample_trade_logs, risk_free_rate=0.0)

        assert isinstance(out, float)
        # Genau wie der Test deines Kumpels prüfen wir hier die mathematischen Grenzen
        # Eine Sharpe Ratio über 100 oder unter -100 ist bei normalen Marktdaten unmöglich
        assert out >= -100.0 and out <= 100.0


    # ========================
    # Edge Case Tests
    # ========================

    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_empty_ticker_data(self, analysis_key, func_name, required_args, sample_trade_logs):
        """
        Edge Case 1: Verify behavior when ticker dataframe is empty.
        """
        func = getattr(analyser_module, func_name)
        empty_ticker = pd.DataFrame(columns=["date", "close"])
        
        out = func(empty_ticker, sample_trade_logs, risk_free_rate=0.0)
        
        assert isinstance(out, float)
        assert out == 0.0

    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_missing_columns(self, analysis_key, func_name, required_args, sample_ticker_data, sample_trade_logs):
        """
        Edge Case 2: Verify behavior when required columns are missing in input dataframes.
        """
        func = getattr(analyser_module, func_name)
        broken_ticker = sample_ticker_data.drop(columns=["close"])
        
        with pytest.raises((KeyError, ValueError)):
            func(broken_ticker, sample_trade_logs, risk_free_rate=0.0)
            
    @pytest.mark.parametrize("analysis_key,func_name,required_args", ANALYSIS_TO_TEST)
    def test_analysis_no_trades_returns_zero(self, analysis_key, func_name, required_args, sample_ticker_data):
        """
        Edge Case 3: Verify behavior when trade logs contain no entries.
        """
        func = getattr(analyser_module, func_name)
        empty_trades = pd.DataFrame(columns=["date", "time", "shares", "type"])
        
        out = func(sample_ticker_data, empty_trades, risk_free_rate=0.0)
        
        assert isinstance(out, float)
        assert out == 0.0
