import pytest
import pandas as pd
from strategies.quant.optimize.validator import Validator

from strategies.quant.tests.test_data_generator import *

# Define all validator pipelines to test with: (pipeline_key, function_name, required_arguments)
VALIDATOR_TO_TEST = [
    ("PIPELINE_V1", "run_validation_pipeline", ["input_data_path", "input_metadata_path", "execute_func", "config_module_name"]),
]


class TestValidator:
    """
    Parametrized test suite for validator.
    
    For each validator function, we test:
    1. Existence of the function
    2. Existence of the function within Validator class
    3. Correctness of the function arguments
    4. Correctness of the output (type, value range)
    """

    @pytest.mark.parametrize("pipeline_key,func_name,required_args", VALIDATOR_TO_TEST)
    def test_validator_function_exists(self, pipeline_key, func_name, required_args):
        """
        Test 1: Verify validator function exists.
        """
        assert hasattr(Validator, func_name)
        assert callable(getattr(Validator, func_name))

    @pytest.mark.parametrize("pipeline_key,func_name,required_args", VALIDATOR_TO_TEST)
    def test_validator_class_structure(self, pipeline_key, func_name, required_args):
        """
        Test 2: Verify helper methods exist in Validator.
        """
        assert hasattr(Validator, "load_market_data")
        assert callable(getattr(Validator, "load_market_data"))
    
    @pytest.mark.parametrize("pipeline_key,func_name,required_args", VALIDATOR_TO_TEST)
    def test_validator_arguments_correctness(self, pipeline_key, func_name, required_args):
        """
        Test 3: Verify validator function accepts all required arguments.
        """
        import inspect
        func = getattr(Validator, func_name)
        sig = inspect.signature(func)
        
        for arg in required_args:
            assert arg in sig.parameters

        @pytest.mark.parametrize("pipeline_key,func_name,required_args", VALIDATOR_TO_TEST)
    def test_validator_output_correctness(self, pipeline_key, func_name, required_args, tmp_path, sample_ticker_data, sample_metadata, dummy_execute_func):
        """
        Test 4: Verify the correctness of the output (type, value range)
        """
        validator = Validator()
        func = getattr(validator, func_name)
        
        # Setup temporary files matching the input data schema
        data_path = tmp_path / "ticker_test.csv"
        meta_path = tmp_path / "meta_test.csv"
        sample_ticker_data.to_csv(data_path, index=False)
        sample_metadata.to_csv(meta_path, index=False)
        
        # KORREKTUR: dummy_execute_func wird jetzt sauber als Fixture injiziert
        out = func(
            input_data_path=str(data_path),
            input_metadata_path=str(meta_path),
            execute_func=dummy_execute_func,
            config_module_name="strategies.quant.configs.active_config"
        )

        assert isinstance(out, float)
        # Verify Sharpe Ratio is within realistic mathematical limits
        assert out >= -100.0 and out <= 100.0


    # ========================
    # Edge Case Tests
    # ========================

    @pytest.mark.parametrize("pipeline_key,func_name,required_args", VALIDATOR_TO_TEST)
    def test_validator_empty_trades_returns_zero(self, pipeline_key, func_name, required_args, tmp_path, sample_ticker_data, sample_metadata):
        """
        Edge Case 1: Verify behavior when no trades are generated in protocol.
        """
        validator = Validator()
        func = getattr(validator, func_name)
        
        # Setup temporary files
        data_path = tmp_path / "ticker_empty_test.csv"
        meta_path = tmp_path / "meta_empty_test.csv"
        sample_ticker_data.to_csv(data_path, index=False)
        sample_metadata.to_csv(meta_path, index=False)
        
        # A dummy execute function that explicitly yields an empty trade history
        def empty_execute_dummy(df, meta, configs):
            return {"trade_df": pd.DataFrame(columns=["date", "time", "shares", "type"])}

        out = func(
            input_data_path=str(data_path),
            input_metadata_path=str(meta_path),
            execute_func=empty_execute_dummy,
            config_module_name="strategies.quant.configs.active_config"
        )

        assert isinstance(out, float)
        assert out == 0.0
