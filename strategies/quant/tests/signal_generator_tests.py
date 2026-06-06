import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from test_data_generator import signal_generation_params, sample_alpha_matrix
from strategies.quant.signal_pipeline.feature_registry import FeatureRegistry
import strategies.quant.signal_pipeline.signal_generator as signal_generator
from strategies.quant.signal_pipeline.signal_generator import generate_signals, CONTEXT_FEATURE_MAP


class TestSignalGenerator:
    """
    Test suite for signal generator.
    
    For the signal generation function, we test:
    1. Existence of the function
    2. Existence of the map/configuration
    3. Correctness of the map/configuration structure
    4. Correctness of the output (type, shape, value range)
    """
    
    # ========================
    # Generate Signals Function Tests
    # ========================
    
    def test_generate_signals_function_exists(self):
        """
        Test 1: Verify generate_signals function exists in signal_generator module.
        """
        assert hasattr(signal_generator, 'generate_signals')
        assert callable(getattr(signal_generator, 'generate_signals'))
    
    def test_context_feature_map_exists(self):
        """
        Test 2: Verify CONTEXT_FEATURE_MAP exists in signal_generator module.
        """
        assert hasattr(signal_generator, 'CONTEXT_FEATURE_MAP')
        assert isinstance(CONTEXT_FEATURE_MAP, dict)
    
    #TODO Create a map test for each context feature
    # The features used in this test are for example only and are not correct
    def test_context_feature_map_correctness(self):
        """
        Test 3: Verify CONTEXT_FEATURE_MAP has correct structure.
        """
        # Check map structure
        entry = CONTEXT_FEATURE_MAP["CONTEXT_FEATURE"]

        assert "func" in entry
        assert "required_parameters" in entry
        
        # Check function reference
        assert callable(entry["func"])
        assert entry["func"] == FeatureRegistry.compute_rsi
        
        # Check required parameters
        assert isinstance(entry["required_parameters"], list)
        assert "rsi_length" in entry["required_parameters"]
    
    def test_generate_signals_output_correctness(self, sample_alpha_matrix, signal_generation_params):
        """
        Test 4: Verify generate_signals output is correct with parameters.
        """
        signal_df = generate_signals(sample_alpha_matrix, signal_generation_params)
        
        # Check output type and shape
        assert isinstance(signal_df, pd.DataFrame)
        assert len(signal_df) == len(sample_alpha_matrix)
        
        # Check output columns
        assert 'alpha_score' in signal_df.columns
        assert 'signal' in signal_df.columns
        
        # Check alpha_score is preserved
        assert signal_df['alpha_score'].equals(sample_alpha_matrix['alpha_score'])
        
        # Check signal values are in [-1, 0, 1]
        assert set(signal_df['signal'].unique()).issubset({-1, 0, 1})

        # Check the output preserves the input index
        assert signal_df.index.equals(sample_alpha_matrix.index)

        # alpha_score should be numeric
        assert pd.api.types.is_numeric_dtype(signal_df['alpha_score'])
        
        # signal should be numeric (int)
        assert pd.api.types.is_numeric_dtype(signal_df['signal'])
    

    # ========================
    # Additional Generate Signals Tests
    # ========================
    """
    1. Test when alpha_score column in the input matrix is empty
    2. Test when alpha_score column in the input matrix is missing
    3. Test when NaN values are within the input dataframe
    4. Test when entries are missing in the parameters
    """

    def test_generate_signals_empty_dataframe(self, signal_generation_params):
        """
        Test generate_signals with empty dataframe.
        """
        df = pd.DataFrame({'alpha_score': []})
        
        signal_df = generate_signals(df, signal_generation_params)
        assert len(signal_df) == 0
    
    def test_generate_signals_missing_alpha_score_column(self, minimal_alpha_matrix, signal_generation_params):
        """
        Test generate_signals fails when alpha_score column is missing.
        """
        df = minimal_alpha_matrix.drop('alpha_score', axis=1)
        
        with pytest.raises(KeyError):
            generate_signals(df, signal_generation_params)
    
    def test_generate_signals_with_nan_values(self, sample_alpha_matrix, signal_generation_params):
        """
        Test generate_signals with NaN values in alpha scores.
        """
        df = sample_alpha_matrix.copy()
        df.loc[df.index[5], 'alpha_score'] = np.nan
        
        signal_df = generate_signals(df, signal_generation_params)
        
        assert isinstance(signal_df, pd.DataFrame)
    
    def test_generate_signals_missing_entry_barrier_param(self, sample_alpha_matrix):
        """
        Test generate_signals fails when entry_barrier parameter is missing.
        """
        params = {'exit_barrier': -0.1}  # Missing entry_barrier
        
        with pytest.raises(KeyError):
            generate_signals(sample_alpha_matrix, params)
    
    def test_generate_signals_missing_exit_barrier_param(self, sample_alpha_matrix):
        """
        Test generate_signals fails when exit_barrier parameter is missing.
        """
        params = {'entry_barrier': 0.8}  # Missing exit_barrier
        
        with pytest.raises(KeyError):
            generate_signals(sample_alpha_matrix, params)
    
    
    # ========================
    # Signal Logic Tests
    # ========================
    
    def test_signal_generation_logic_buy_signal(self, signal_generation_params):
        """
        Test that alpha scores above entry_barrier generate buy signal (1).
        """
        matrix = pd.DataFrame({
            'alpha_score': [0.85, 0.99, 1.0]
        })
        # entry_barrier = 0.8
        signal_df = generate_signals(matrix, signal_generation_params)
        
        assert (signal_df['signal'] == 1).all()
    
    def test_signal_generation_logic_sell_signal(self, signal_generation_params):
        """
        Test that alpha scores below exit_barrier generate sell signal (-1).
        """
        matrix = pd.DataFrame({
            'alpha_score': [-0.2, -0.5, -1.0]
        })        
        # exit_barrier = -0.1
        signal_df = generate_signals(matrix, signal_generation_params)
        
        assert (signal_df['signal'] == -1).all()
    
    def test_signal_generation_logic_hold_signal(self, signal_generation_params):
        """
        Test that alpha scores between barriers generate hold signal (0).
        """
        matrix = pd.DataFrame({
            'alpha_score': [0.5, 0.0, -0.05]
        })     
        # entry_barrier = 0.8
        # exit_barrier = -0.1
        signal_df = generate_signals(matrix, signal_generation_params)
        
        assert (signal_df['signal'] == 0).all()

