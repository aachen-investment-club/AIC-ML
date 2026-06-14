from collections.abc import Callable
import pandas as pd

class Backtester():
    def __init__(self, input_data: pd.DataFrame, input_metadata: pd.DataFrame, 
                 execute: Callable[[pd.DataFrame, pd.DataFrame, dict], dict],
                 configs: dict):
        self._input_data = input_data
        self._input_metadata = input_metadata
        self._execute = execute
        self._configs = configs
        self._trade_logs = {}

    def backtest(self) -> dict:
        self._trade_logs = self._execute(self._input_data, self._input_metadata, self._configs)

        return self._trade_logs
    
    def set_input_data(self, input_data: pd.DataFrame):
        self._input_data = input_data

    def set_input_metadata(self, input_metadata: pd.DataFrame):
        self._input_metadata = input_metadata

    def set_execute(self, execute: Callable[[pd.DataFrame, pd.DataFrame], dict]):
        self._execute = execute

    def set_configs(self, configs: dict):
        self._configs = configs

    def get_trade_logs(self):
        return self._trade_logs