from collections.abc import Callable
import pandas as pd
import copy
class Backtester():
    def __init__(self, input_data: pd.DataFrame, input_metadata: pd.DataFrame, 
                 execute: Callable[[pd.DataFrame, pd.DataFrame, dict], dict],
                 configs: dict):
        self._input_data = input_data
        self._input_metadata = input_metadata
        self._execute = execute
        self._configs = copy.deepcopy(configs)
        self._trade_logs = {}

    def backtest(self) -> dict:
        self._trade_logs = self._execute(self._input_data, self._input_metadata, self._configs)

        return self._trade_logs
    
    def set_input_data(self, input_data: pd.DataFrame) -> None:
        self._input_data = input_data

    def set_input_metadata(self, input_metadata: pd.DataFrame) -> None:
        self._input_metadata = input_metadata

    def set_execute(self, execute: Callable[[pd.DataFrame, pd.DataFrame, dict], dict]) -> None:
        self._execute = execute

    def set_configs(self, configs: dict) ->None:
        self._configs = copy.deepcopy(configs)

    def get_trade_logs(self) ->dict:
        return self._trade_logs