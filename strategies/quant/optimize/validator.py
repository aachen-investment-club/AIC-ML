from collections.abc import Callable
import pandas as pd
from strategies.quant.optimize.backtester import Backtester #simuliert strategie
from strategies.quant.optimize.analyser import analyser #gibt sharpe ratio
from strategies.quant.optimize.optimizer import run_optimizer # versucht alle parameter für eine strategie und gibt die beste
#load data -> optimize -> simulate -> analyze
class Validator:
    def load_market_data(self, file_path:str) -> pd.DataFrame:

        return pd.read_csv(file_path)#nimmt datei von preisen 


    def run_validation_pipeline(self, input_data_path: str, input_metadata_path: str, execute_func: Callable, config_module_name: str) ->float:
    #metadata ist wie zb BTC ETH oder so und data ist datum und preis

        input_data = self.load_market_data(input_data_path)
        input_metadata = self.load_market_data(input_metadata_path)

    #optimizer looks for best parameters

        optimized_parameters = run_optimizer(input_data_path=input_data_path, input_metadata_path=input_metadata_path, n_trials=50,config_module_name=config_module_name) 
        btester = Backtester(input_data=input_data, input_metadata=input_metadata, execute=execute_func, configs=optimized_config)

    #generate trade logs
        trade_logs_dict = btester.backtest()
        trade_df = trade_logs_dict.get("trade_df", pd.DataFrame()) #something for security/test

        if trade_df.empty:
            print("No trades in protocol. Sharpe Rati is 0.0")
            return 0.0

        sharpe_ratio = analyse(input_data=input_data,trade_logs=trade_df, risk_free_rate=0.0)

        return float(sharpe_ratio)


        


        

        
        

