import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def feature_params():
    """
    Standard feature parameters
    """
    return {
        "rsi_length": 14,

        "ma_fast": 5, 
        "ma_slow": 20,
    }


@pytest.fixture
def signal_generation_params():
    """
    Standard signal generation parameters.
    """
    return {
        'entry_barrier': 0.8,
        'exit_barrier': -0.1
    }


@pytest.fixture
def sample_ohlcv_data():
    """
    Create sample OHLCV data for testing.
    """
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.2,
        'high': close_prices + abs(np.random.randn(100) * 0.3),
        'low': close_prices - abs(np.random.randn(100) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return df


@pytest.fixture
def sample_alpha_matrix():
    """
    Generate sample feature matrix with alpha scores.
    """
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    alpha_matrix = pd.DataFrame({
        'ticker': ['AAPL'] * 100,
        'date': dates,
        'alpha_score': np.random.uniform(-1.0, 1.0, 100),
        'MA_SPREAD': np.random.uniform(-0.5, 0.5, 100),
        'RSI': np.random.uniform(-50, 50, 100)
    })
    return alpha_matrix


@pytest.fixture
def sample_signal_matrix(signal_generation_params):
    """
    Generate sample matrix with signals depending on the config parameters
    """
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    signal_matrix = pd.DataFrame({
        'ticker': ['AAPL'] * 100,
        'date': dates,
        'alpha_score': np.random.uniform(-1.0, 1.0, 100),
    })

    signals = []
    for _, row in signal_matrix.iterrows():
        alpha_score = row['alpha_score']
        
        if alpha_score >= signal_generation_params["entry_barrier"]:
            signals.append(1)
        elif alpha_score <= signal_generation_params["exit_barrier"]:
            signals.append(-1)
        else:
            signals.append(0)
    signal_matrix["signal"] = signals
    return signal_matrix