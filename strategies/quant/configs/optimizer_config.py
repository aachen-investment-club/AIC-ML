# Set the bounds for optimization
PARAMETER_BOUNDS = {
    # Alpha parameters
    "rsi_length": {"type": "int", "low": 5, "high": 30},
    "ma_fast": {"type": "int", "low": 5, "high": 50},
    "ma_slow": {"type": "int", "low": 40, "high": 200},

    # Context parameters
    "entry_barrier": {"type": "float", "low": 0.5, "high": 1.0},
    "exit_barrier": {"type": "float", "low": -1.0, "high": 0.0},
}