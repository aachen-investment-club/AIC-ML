import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from strategies.quant.optimize import optimizer as optimizer_module
from strategies.quant.optimize.optimizer import run_optimizer

from strategies.quant.tests.test_data_generator import *


# Define all optimizer functions to test with: (optimizer_key, function_name, required_arguments)
OPTIMIZER_TO_TEST = [
    ("OPTUNA_OPTIMIZER", "run_optimizer", ["input_data_path", "input_metadata_path", "n_trials", "config_module_name"]),
]


class TestOptimizer:
    """
    Parametrized test suite for optimizer.
    
    For each optimizer function, we test:
    1. Existence of the function
    2. Existence of the function within the module
    3. Correctness of the function arguments
    4. Correctness of the output (type, structure)
    """

    @pytest.mark.parametrize("optimizer_key,func_name,required_args", OPTIMIZER_TO_TEST)
    def test_optimizer_function_exists(self, optimizer_key, func_name, required_args):
        """
        Test 1: Verify optimizer function exists.
        """
        assert hasattr(optimizer_module, func_name)
        assert callable(getattr(optimizer_module, func_name))

    @pytest.mark.parametrize("optimizer_key,func_name,required_args", OPTIMIZER_TO_TEST)
    def test_optimizer_map_exists(self, optimizer_key, func_name, required_args):
        """
        Test 2: Verify optimizer entry is active in module namespace.
        """
        # Da der Optimizer keine Registry-Map wie die Features nutzt,
        # prüfen wir hier die Bindung im Modul-Namespace
        assert func_name in dir(optimizer_module)
    
    @pytest.mark.parametrize("optimizer_key,func_name,required_args", OPTIMIZER_TO_TEST)
    def test_optimizer_arguments_correctness(self, optimizer_key, func_name, required_args):
        """
        Test 3: Verify optimizer function accepts all required arguments.
        """
        import inspect
        func = getattr(optimizer_module, func_name)
        sig = inspect.signature(func)
        
        for arg in required_args:
            assert arg in sig.parameters

    @pytest.mark.parametrize("optimizer_key,func_name,required_args", OPTIMIZER_TO_TEST)
    def test_optimizer_output_correctness(self, optimizer_key, func_name, required_args, tmp_path, sample_ticker_data, sample_metadata):
        """
        Test 4: Verify the correctness of the output (type, structure)
        """
        func = getattr(optimizer_module, func_name)
        
        # Setup temporary files matching the input data schema
        data_path = tmp_path / "ticker_opt_test.csv"
        meta_path = tmp_path / "meta_opt_test.csv"
        sample_ticker_data.to_csv(data_path, index=False)
        sample_metadata.to_csv(meta_path, index=False)
        
        # Wir faken den Optuna-Lauf mit einem Mock, damit der Unit Test schnell bleibt (1 Trial)
        out = func(
            input_data_path=str(data_path),
            input_metadata_path=str(meta_path),
            n_trials=1,
            config_module_name="strategies.quant.configs.active_config"
        )

        assert isinstance(out, dict)
        assert "active_strategy" in out
        assert "alpha_parameters" in out
        assert "context_parameters" in out


    # ========================
    # Edge Case Tests
    # ========================

    @pytest.mark.parametrize("optimizer_key,func_name,required_args", OPTIMIZER_TO_TEST)
    def test_optimizer_invalid_module_path(self, optimizer_key, func_name, required_args, tmp_path, sample_ticker_data, sample_metadata):
        """
        Edge Case 1: Verify behavior when the config module path does not exist.
        """
        func = getattr(optimizer_module, func_name)
        data_path = tmp_path / "ticker_opt_err.csv"
        meta_path = tmp_path / "meta_opt_err.csv"
        sample_ticker_data.to_csv(data_path, index=False)
        sample_metadata.to_csv(meta_path, index=False)
        
        with pytest.raises(ModuleNotFoundError):
            func(
                input_data_path=str(data_path),
                input_metadata_path=str(meta_path),
                n_trials=1,
                config_module_name="this.module.does.not.exist"
            )

    @pytest.mark.parametrize("optimizer_key,func_name,required_args", OPTIMIZER_TO_TEST)
    def test_optimizer_file_not_found(self, optimizer_key, func_name, required_args):
        """
        Edge Case 2: Verify behavior when input data paths are invalid.
        """
        func = getattr(optimizer_module, func_name)
        
        with pytest.raises(FileNotFoundError):
            func(
                input_data_path="invalid_path_to_data.csv",
                input_metadata_path="invalid_path_to_meta.csv",
                n_trials=1,
                config_module_name="strategies.quant.configs.active_config"
            )
