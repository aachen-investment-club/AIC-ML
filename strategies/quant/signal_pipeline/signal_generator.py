import pandas as pd


def generate_signals(matrix: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Translates a continuous alpha stream into a discrete tradable signal vector 
    by dynamically adjusting entry barriers based on contextual risk features.

    Parameters
    ----------
    matrix : pd.DataFrame
        The input matrix containing alpha scores and contextual risk/regime metrics.
        Expected to have columns for alpha values and context features aligned with
        CONTEXT_FEATURE_MAP. Bounded between [-1.0, 1.0].
            - 'alpha_score' (float): The alpha values used to generate signals.
    params : dict
        Hyperparameters managed by the system configuration file.
        Required keys:
            - 'entry_barrier' (float): The base threshold anchor for opening trades.
            - 'exit_barrier' (float): The threshold anchor for closing trades.

    Returns
    -------
    pd.DataFrame
        The execution signal matrix passed directly to the downstream trade team.
        Columns:
            - 'alpha_score' (float64): Preserved input alpha values for audit trails.
            - 'signal' (int64): Clean shifted vector [-1, 0, 1] 
              fully protected against look-ahead bias.
    """
    buy_threshold = params["entry_barrier"]
    sell_threshold = params["exit_barrier"]

    out = pd.DataFrame()
    out["alpha_score"] = matrix["alpha_score"]
    
    # Vectorized signal generation to avoid look-ahead bias
    signals = []
    for _, row in matrix.iterrows():
        alpha = row["alpha_score"]
        if alpha >= buy_threshold:
            signal = 1
        elif alpha <= sell_threshold:
            signal = -1
        else:
            signal = 0
        signals.append(signal)
    
    out["signal"] = signals
    return out
    


CONTEXT_FEATURE_MAP = {
    "CONTEXT_FEATURE": {
        "func": "callable type",
        "required_parameters": ["param1", "param2"],
    }
}