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


@pytest.fixture
def sample_ticker_data():
    
    dates = pd.date_range(start="2026-01-01", periods=50, freq="D")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(50) * 1.5)
    
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": prices
    })

@pytest.fixture
def sample_trade_logs():
    
    return pd.DataFrame({
        "date": ["2026-01-05", "2026-01-15", "2026-01-25"],
        "time": ["09:30:00", "14:15:00", "11:00:00"],
        "shares": [10.0, 5.0, 8.0],
        "type": ["PURCHASE", "SALE", "PURCHASE"]
    })

@pytest.fixture
def sample_metadata():
    
    return pd.DataFrame({
        "ticker": ["AAPL"],
        "asset_class": ["EQUITY"]
    })

@pytest.fixture
def sample_configs():
    
    return {
        "active_strategy": "TREND_FOLLOWING",
        "alpha_parameters": {
            "rsi_length": 14,
            "ma_fast": 5
        },
        "context_parameters": {
            "entry_barrier": 0.8
        }
    }

@pytest.fixture
def dummy_execute_func():
    
    def _execute(input_data, input_metadata, configs):
        # Gibt ein gültiges Dictionary mit Trade-Logs zurück
        return {
            "trade_df": pd.DataFrame({
                "date": ["2026-01-05", "2026-01-15"],
                "time": ["09:30:00", "14:15:00"],
                "shares": [10.0, 5.0],
                "type": ["PURCHASE", "SALE"]
            })
        }
    return _execute

@pytest.fixture
def sample_ticker_data():
    """
    Erzeugt künstliche Kursdaten für die analyse-Funktion.
    """
    dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": [100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 110.0, 108.0, 112.0, 115.0]
    })


@pytest.fixture
def sample_trade_logs():
    """
    Erzeugt künstliche Trade-Einträge für den Portfolio-Analyser.
    """
    return pd.DataFrame({
        "date": ["2026-01-02", "2026-01-05"],
        "time": ["10:00:00", "14:30:00"],
        "shares": [10.0, 5.0],
        "type": ["PURCHASE", "SALE"]
    })
