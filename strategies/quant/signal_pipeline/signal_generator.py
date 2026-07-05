import pandas as pd
import numpy as np


def generate_signals(matrix: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Translates a continuous alpha stream into a discrete tradable signal vector
    by dynamically adjusting entry barriers based on the Hurst Exponent regime.

    Parameters
    ----------
    matrix : pd.DataFrame
        Required columns:
            - 'alpha_score' (float): The raw, continuous alpha metric.
            - 'hurst_exponent' (float): Computed Hurst series from FeatureRegistry.
    params : dict
        Required keys:
            - 'entry_barrier' (float): Base threshold anchor for long entry (e.g., 0.3).
            - 'exit_barrier' (float): Base threshold anchor for short entry (e.g., -0.3).
            - 'alpha_smooth_span' (int): Window span for smoothing alpha (EMA).
            - 'regime_sensitivity' (float): Controls how aggressively Hurst modifies barriers.

    Returns
    -------
    pd.DataFrame
        Columns: ['alpha_score', 'smoothed_alpha', 'hurst_exponent', 'signal']
    """
    buy_threshold = params["entry_barrier"]
    sell_threshold = params["exit_barrier"]

    out = pd.DataFrame()
    out["alpha_score"] = matrix["alpha_score"]
    out["hurst_exponent"] = matrix["hurst_exponent"]
    out["smoothed_alpha"] = (
        matrix["alpha_score"].ewm(span=params["alpha_smooth_span"], adjust=False).mean()
    )

    # Dynamic thresholding using Hurst Exponent
    hurst_deviation = 0.5 - out["hurst_exponent"]
    scaling_factor = np.exp(params["regime_sensitivity"] * hurst_deviation)
    dynamic_buy_threshold = params["entry_barrier"] * scaling_factor
    dynamic_sell_threshold = params["exit_barrier"] * scaling_factor
    
    # Vectorized signal generation to avoid look-ahead bias
    conditions = [
        out["smoothed_alpha"] >= dynamic_buy_threshold,
        out["smoothed_alpha"] <= dynamic_sell_threshold,
    ]
    choices = [1, -1]

    # Default value is 0 if no conditions are met
    out["signal"] = np.select(conditions, choices, default=0)

    # Shift the signal by 1 bar to strictly protect against look-ahead bias
    # (An alpha at market close can only be traded on the next open bar)
    out["signal"] = out["signal"].shift(1).fillna(0).astype(np.int64)
    
    return out
    

@staticmethod
def compute_hurst_exponent(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Calculates a rolling Hurst Exponent to identify market regimes.
    
    Interpretation:
    - H < 0.5 : Mean-reverting (Anti-persistent)
    - H = 0.5 : Random Walk (Brownian Motion)
    - H > 0.5 : Trending (Persistent)

    Parameters
    ----------
    df : pd.DataFrame
        Required column: 'close' (float64)
    params : dict
        Required keys:
            - 'hurst_window' (int): The rolling lookback window.
            - 'max_lag' (int): Maximum time lag for scaling (typically 20-50).

    Returns
    -------
    pd.Series
        Rolling Hurst Exponent values mapped to the original index.
    """
    window = params['hurst_window']
    max_lag = params.get('max_lag', 20)

    if max_lag >= window:
        max_lag = max(5, window // 4)

    lags = np.arange(2, max_lag)

    def _calculate_hurst(window_series):
        # Avoid execution if the window has NaNs or insufficient data
        if len(window_series) < window or np.any(np.isnan(window_series)):
            return np.nan
        
        # Calculate variance of differences for each lag
        tau = []
        for lag in lags:
            diff = window_series[lag:] - window_series[:-lag]
            tau.append(np.std(diff))
        
        # Prevent log(0) issues if variance is zero
        tau = np.array(tau)
        valid = tau > 0
        if not np.any(valid):
            return np.nan
            
        # Perform linear regression on log-log scale
        poly = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
        
        # The slope is equal to H
        return poly[0]

    # Use raw=True for optimal rolling performance in pandas
    hurst_values = (
        df['close']
        .rolling(window=window)
        .apply(_calculate_hurst, raw=True)
    )
    
    return pd.Series(hurst_values, index=df.index)


CONTEXT_FEATURE_MAP = {
    "HURST_EXPONENT": {
        "func": compute_hurst_exponent,
        "required_parameters": ["hurst_window", "max_lag"]
    }
}