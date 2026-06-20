import pandas as pd
import optuna
import importlib
import copy

from strategies.quant.configs.optimizer_config import PARAMETER_BOUNDS
from strategies.quant.signal_pipeline.execute import execute
from strategies.quant.signal_pipeline.alpha_generator import STRATEGY_MAP
from strategies.quant.signal_pipeline.feature_registry import FEATURE_MAP
from strategies.quant.signal_pipeline.signal_generator import CONTEXT_FEATURE_MAP
from strategies.quant.optimize.backtester import Backtester


def objective(
        trial: optuna.Trial, 
        base_config: dict, 
        input_data: pd.DataFrame, 
        input_metadata: pd.DataFrame
    ) -> float:
    
    # Clone the base config
    trial_config = copy.deepcopy(base_config)

    # Determine the current used strategy
    strategy = trial_config["active_strategy"]
    required_features = STRATEGY_MAP[strategy]["required_features"]

    # Find and optimize the required alpha feature parameters
    for feature_key in required_features:
        required_params = FEATURE_MAP[feature_key]["required_parameters"]

        for param in required_params:
            if param in PARAMETER_BOUNDS:
                bounds = PARAMETER_BOUNDS[param]

                if bounds["type"] == "int":
                    trial_config["alpha_parameters"][param] = trial.suggest_int(
                        param, bounds["low"], bounds["high"]
                    )
                elif bounds["type"] == "float":
                    trial_config["alpha_parameters"][param] = trial.suggest_float(
                        param, bounds["low"], bounds["high"]
                    )

    # Find and optimize the required context feature parameters
    for context_feature_key in CONTEXT_FEATURE_MAP:
        required_params = CONTEXT_FEATURE_MAP[context_feature_key]["required_parameters"]

        for param in required_params:
            if param in PARAMETER_BOUNDS:
                bounds = PARAMETER_BOUNDS[param]

                if bounds["type"] == "int":
                    trial_config["context_parameters"][param] = trial.suggest_int(
                        param, bounds["low"], bounds["high"]
                    )
                elif bounds["type"] == "float":
                    trial_config["context_parameters"][param] = trial.suggest_float(
                        param, bounds["low"], bounds["high"]
                    )

    # Instantiate and run the backtester
    tester = Backtester(
        input_data=input_data, 
        input_metadata=input_metadata, 
        execute=execute,
        configs=trial_config
    )
    
    # 4. Run the simulation
    trade_logs = tester.backtest()

    # 5. Get the performance
    performance = 0
    
    # 5. Extract target performance metric (assumed calculated inside your output_adapter)
    return performance


def run_optimizer(
        input_data_path: str, 
        input_metadata_path: str, 
        n_trials: int = 50,
        config_module_name: str = "strategies.quant.configs.active_config"
    ) -> dict:
    """
        Runs the Optuna optimizer to optimize the active config's parameter values
    """

    # Load the data
    df_data = pd.read_csv(input_data_path)
    df_meta = pd.read_csv(input_metadata_path)

    # Input the active config
    config_module = importlib.import_module(config_module_name)
    importlib.reload(config_module) # Force reload in case it changed in-session
    starting_config = copy.deepcopy(config_module.ACTIVE_CONFIG)

    # Setup and run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, starting_config, df_data, df_meta),
        n_trials=n_trials,
    )

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Performance Value: {study.best_trial.value:.4f}")

    # Map Optuna's flat winning parameters back into config
    best_config = copy.deepcopy(starting_config)
    for param_name, winning_value in study.best_trial.params.items():
        if param_name in best_config["alpha_parameters"].keys():
            best_config["alpha_parameters"][param_name] = winning_value
        elif param_name in best_config["context_parameters"].keys():
            best_config["context_parameters"][param_name] = winning_value

    # Overwrite the active_config.py file with the updated dictionary
    config_file_path = f"{config_module_name.replace('.', '/')}.py"
    with open(config_file_path, "w") as f:
        # Use repr() to write the raw python dictionary format out cleanly
        f.write(f"ACTIVE_CONFIG = {repr(best_config)}\n")
    
    print(f"Successfully wrote optimized parameters back to {config_file_path}!")
    return best_config