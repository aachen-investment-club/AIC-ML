ACTIVE_CONFIG = {
    # Strategy choice
    "active_strategy": "TREND_FOLLOWING",

    # Alpha feature parameters
    "alpha_parameters": {
        "rsi_length": 14,

        "ma_fast": 20,
        "ma_slow": 50
    },

    # Context feature parameters 
    "context_parameters": {
        # Thresholds (between -1.0 and 1.0)
        "entry_barrier": 0.8,
        "exit_barrier": -0.1
    }
}