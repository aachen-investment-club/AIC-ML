import pandas as pd
import pandas-ta as ta
import numpy as np


class FeatureRegistry:

    @staticmethod
    def compute_rsi(df: pd.DataFrame, params: dict) -> pd.Series:
        """
        Calculates a zero-centered, normalized Relative Strength Index (RSI).

        Parameters
        ----------
        df : pd.DataFrame
            The raw input market data matrix. Must conform to the input schema.
            Required column:
                - 'close' (float64): Closing price of the asset.
        params : dict
            Hyperparameters managed by the system configuration.
            Required key:
                - 'rsi_length' (int): The lookback window for the RSI smoothing.

        Returns
        -------
        pd.Series
            A series containing the feature data.
            RSI values shifted to oscillate between -50.0 (oversold boundary) and +50.0 (overbought boundary).
        """
        rsi_array = ta.rsi(df['close'], length=params['rsi_length']) - 50
        rsi_series = pd.Series(rsi_array, index=df.index)
        return rsi_series
    

    @staticmethod
    def compute_ma_spread(df: pd.DataFrame, params: dict) -> pd.Series:
        pass
    

    @staticmethod
    def compute_log_returns(df: pd.DataFrame, params: dict) -> pd.Series:
        pass


    @staticmethod
    def compute_zscore_distance_rolling_mean(df: pd.DataFrame, params: dict) -> pd.Series:
        pass


    @staticmethod
    def compute_rolling_volatility(df: pd.DataFrame, params: dict) -> pd.Series:
        pass


FEATURE_MAP = {
    "RSI": {
        "func": FeatureRegistry.compute_rsi,
        "required_parameters": ["rsi_length"]
    },
    "MA_SPREAD": {
        "func": FeatureRegistry.compute_ma_spread,
        "required_parameters": ["ma_fast", "ma_slow"]
    }
}