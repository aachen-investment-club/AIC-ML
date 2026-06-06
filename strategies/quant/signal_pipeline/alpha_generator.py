import pandas as pd
from strategies.quant.signal_pipeline.feature_registry import FeatureRegistry


def trend_following_alpha(feature_matrix: pd.DataFrame) -> pd.Series:
    """
    Generates an Alpha Score based on structural geometric trend expansions.

    Mathematical Thesis:
        Asset prices maintain multi-period momentum. When a short-term trend line 
        diverges significantly above a long-term structural baseline, macro buy 
        liquidity is entering, yielding positive directional alpha.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        The data matrix enriched by Teammate 1.
        Required column:
            - 'ma_spread' (float64): Moving Average cross-distance percentages.

    Returns
    -------
    pd.Series
        A continuous vector stream representing directional market pressure.
        Shape matches input index. Bounded strictly between [-1.0, 1.0] via truncation.
        Values:
            - Close to +1.0: Strong upward structural momentum.
            - Close to  0.0: Moving Averages tangled / Sideways trendless noise.
            - Close to -1.0: Strong downward structural momentum.
    """
    spread = feature_matrix['MA_SPREAD']
    alpha = spread * 10.0
    return alpha.clip(-1.0, 1.0)


def rsi_mean_reversion_alpha(feature_matrix: pd.DataFrame) -> pd.Series:
    return pd.Series()


STRATEGY_MAP = {
    "MEAN_REVERSION": {
        "alpha_func": rsi_mean_reversion_alpha,
        "required_features": ["RSI"]
    },
    "TREND_FOLLOWING": {
        "alpha_func": trend_following_alpha,
        "required_features": ["MA_SPREAD"]
    }
}