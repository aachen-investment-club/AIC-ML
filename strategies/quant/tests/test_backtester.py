import pytest
import pandas as pd
import copy
from strategies.quant.optimize.backtester import Backtester
from test_data_generator import sample_ticker_data, sample_metadata, dummy_execute_func, sample_configs


# Define all backtester implementations to test with: (backtester_key, class_name, required_methods)
BACKTESTER_TO_TEST = [
    ("BASE_BACKTESTER", "Backtester", ["backtest", "set_input_data", "set_input_metadata", "set_execute", "set_configs", "get_trade_logs"]),
]


class TestBacktester:
    """
    Parametrized test suite for backtester.
    
    For each backtester class, we test:
    1. Existence of the class
    2. Existence of all required methods
    3. Correctness of setter and getter mechanics
    4. Correctness of the output (type, structure)
    """

    @pytest.mark.parametrize("backtester_key,class_name,required_methods", BACKTESTER_TO_TEST)
    def test_backtester_class_exists(self, backtester_key, class_name, required_methods):
        """
        Test 1: Verify backtester class exists.
        """
        import strategies.quant.optimize.backtester as bt_module
        assert hasattr(bt_module, class_name)
        assert isinstance(getattr(bt_module, class_name), type)

    @pytest.mark.parametrize("backtester_key,class_name,required_methods", BACKTESTER_TO_TEST)
    def test_backtester_methods_exist(self, backtester_key, class_name, required_methods):
        """
        Test 2: Verify all required interface methods exist in the class.
        """
        for method in required_methods:
            assert hasattr(Backtester, method)
            assert callable(getattr(Backtester, method))
    
    @pytest.mark.parametrize("backtester_key,class_name,required_methods", BACKTESTER_TO_TEST)
    def test_backtester_state_mutators(self, backtester_key, class_name, required_methods, sample_ticker_data, sample_metadata, dummy_execute_func, sample_configs):
        """
        Test 3: Verify setter and getter methods correctly update internal state.
        """
        tester = Backtester(
            input_data=sample_ticker_data, 
            input_metadata=sample_metadata, 
            execute=dummy_execute_func, 
            configs=sample_configs
        )
        
        # Test alternative inputs
        alt_data = sample_ticker_data.copy()
        tester.set_input_data(alt_data)
        assert tester._input_data.equals(alt_data)
        
        alt_configs = copy.deepcopy(sample_configs)
        tester.set_configs(alt_configs)
        assert tester._configs == alt_configs

    @pytest.mark.parametrize("backtester_key,class_name,required_methods", BACKTESTER_TO_TEST)
    def test_backtester_output_correctness(self, backtester_key, class_name, required_methods, sample_ticker_data, sample_metadata, dummy_execute_func, sample_configs):
        """
        Test 4: Verify the correctness of the output (type, shape).
        """
        tester = Backtester(
            input_data=sample_ticker_data, 
            input_metadata=sample_metadata, 
            execute=dummy_execute_func, 
            configs=sample_configs
        )
        
        out = tester.backtest()

        assert isinstance(out, dict)
        assert tester.get_trade_logs() == out


    # ========================
    # Edge Case Tests
    # ========================

    @pytest.mark.parametrize("backtester_key,class_name,required_methods", BACKTESTER_TO_TEST)
    def test_backtester_config_deepcopy_isolation(self, backtester_key, class_name, required_methods, sample_ticker_data, sample_metadata, dummy_execute_func, sample_configs):
        """
        Edge Case 1: Verify behavior when external config dictionary is mutated after initialization.
        """
        mutable_config = copy.deepcopy(sample_configs)
        tester = Backtester(
            input_data=sample_ticker_data, 
            input_metadata=sample_metadata, 
            execute=dummy_execute_func, 
            configs=mutable_config
        )
        
        # Mutate the configuration dictionary externally
        mutable_config["alpha_parameters"]["corrupted_key"] = 999
        
        # Verify the internal state of the Backtester is deeply isolated
        assert "corrupted_key" not in tester._configs["alpha_parameters"]
