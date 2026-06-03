import pandas as pd
import pandas_ta as ta
import numpy as np


class FeatureRegistry:

    @staticmethod
    def compute_rsi(df: pd.DataFrame, params: dict) -> pd.DataFrame:
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
        pd.DataFrame
            A single-column DataFrame sharing the input index.
            Column:
                - 'rsi_normalized' (float64): RSI values shifted to oscillate 
                  between -50.0 (oversold boundary) and +50.0 (overbought boundary).
        """
        out = pd.DataFrame(index=df.index)
        out['rsi_normalized'] = ta.rsi(df['close'], length=params['rsi_length']) - 50
        return out
    

    @staticmethod
    def compute_ma_spread(df: pd.DataFrame, params: dict):
        pass
    

    @staticmethod
    def compute_log_returns(df: pd.DataFrame, params: dict):
        pass


    @staticmethod
    def compute_zscore_distance_rolling_mean(df: pd.DataFrame, params: dict):
        pass


    @staticmethod
    def compute_rolling_volatility(df: pd.DataFrame, params: dict):
        pass