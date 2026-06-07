
from ...strategy import Strategy
from ...interfaces import Security, Currency, TransactionType
from .env import TradingEnv



from portfolio.models.features import Features
from portfolio.models.metrics import Metrics 
from portfolio.models.market import Market


import os
import numpy as np
import pandas as pd
import joblib
import json
from typing import Optional
from datetime import date, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env



class RLStrategy(Strategy): 

    strategy_name = "RL_PPO_strategy" 
    strategy_file_name = strategy_name + ".json"
    explanation = "first RL strategy" 
    tickers = ["INTC", "MSFT", "NVDA"]
    train_start = "2015-01-01" #: trained on bullish market. 
    train_end   = "2022-01-01"
    window = 10

    eval_start = "2022-01-01" 
    eval_end = "2023-01-01"


    @classmethod
    def get_model_extension(cls) -> str:
        return ".zip"

    @classmethod
    def _save_model_to_disk(cls, path: str):
        cls.model.save(path)



    @classmethod
    def _load_model_from_disk(cls, path: str):
        print(f"Loading from {path}")
        cls.model = PPO.load(path)

    @classmethod
    def get_data_for_trade(cls, ticker: str = 'AAPL', target_date=None) -> pd.DataFrame:
        from datetime import datetime

        today = datetime.now()- timedelta(days = 1)

        query_result = Market.get_historical_data(
                cls.tickers, 
                start = today-timedelta(days = cls.window+100),   #: TODO this is a heristic for weekends!
                end =today
            )
        query_result= query_result[['ticker', 'date', 'price_close']]
        query_result.rename(columns ={"ticker":"Ticker", "date": "Date"}, inplace = True)
        query_result= query_result.set_index("Date")
        cls.trade_data  = {
            ticker: query_result[query_result["Ticker"]==ticker]["price_close"] for ticker in cls.tickers
        }

    
    @classmethod
    def extract_features_for_trade(cls, trade_df: pd.DataFrame) -> np.ndarray:
        pass

    @classmethod
    def _execute_trade(cls,  target_date=None) -> int:
        from datetime import datetime
        cls.get_data_for_trade()
        cls.env = TradingEnv(cls.trade_data, randomize_start=False, window = cls.window)
        obs, _ = cls.env.reset()
        done = False
        final_action = None
        today = datetime.today()
        while not done:
            action, _ = cls.model.predict(obs, deterministic=True)
            obs, _, done, _, info = cls.env.step(action)
            if done: 
                print(action)
                final_action = action

        if final_action is not None: 
            for ticker_index, action in enumerate(final_action): 
                if action == TradingEnv.BUY: 
                    cls.tradelog.append_trade(
                        type = TransactionType.BUY, 
                        currency = Currency.USD, 
                        date = str(today), 
                        shares =1.0, 
                        security = Security(
                            name = cls.tickers[ticker_index], 
                            ticker= cls.tickers[ticker_index],
                            currency = Currency.USD
                        )
                    )
                    print(f"BUY: {cls.tickers[ticker_index]}")


                elif action == TradingEnv.SELL: 
                    cls.tradelog.append_trade(
                        type = TransactionType.SELL, 
                        currency = Currency.USD, 
                        date = str(today), 
                        shares =1.0, 
                        security = Security(
                            name = cls.tickers[ticker_index], 
                            ticker= cls.tickers[ticker_index],
                            currency = Currency.USD
                        )
                    )
                    print(f"SELL: {cls.tickers[ticker_index]}")

        print(cls.tradelog)
        return 1 #: this is just a junk return value, no meaning. 



    @classmethod
    def _execute_test(cls, model_version: Optional[str] = None) -> dict: 

        cls.env = TradingEnv(cls.test_data, randomize_start=False, window = cls.window)
        obs, _ = cls.env.reset()
        done = False
        lines = []
        while not done:
            action, _ = cls.model.predict(obs, deterministic=True)
            obs, _, done, _, info = cls.env.step(action)
            report = cls.env.render()
            lines.append(report)

        final = info["portfolio_value"]

        baseline = sum(
            cls.test_data[t].iloc[-1] / cls.test_data[t].iloc[0] * (10_000 / len(cls.tickers))
            for t in cls.tickers 
        )

        cls.get_manager().save_lines(lines, "render_results.txt", version_folder = model_version)

        print(f"\nFinal portfolio : ${final:,.2f}")
        print(f"Equal-weight buy-and-hold : ${baseline:,.2f}")
        return {
            "final_portfolio": final, 
            "buy-and-hold_benchmark": baseline
        }





    @classmethod
    def get_training_data(cls):

        from datetime import datetime, timedelta
        
        query_result = Market.get_historical_data(
                cls.tickers, 
                start = cls.train_start,  
                end =cls.train_end 
            )
        query_result= query_result[['ticker', 'date', 'price_close']]
        query_result.rename(columns ={"ticker":"Ticker", "date": "Date"}, inplace = True)
        query_result= query_result.set_index("Date")
        cls.train_data  = {
            ticker: query_result[query_result["Ticker"]==ticker]["price_close"] for ticker in cls.tickers
        }
        query_result = Market.get_historical_data(
                cls.tickers, 
                start = cls.eval_start,  
                end =cls.eval_end 
            )
        query_result= query_result[['ticker', 'date', 'price_close']]
        query_result.rename(columns ={"ticker":"Ticker", "date": "Date"}, inplace = True)
        query_result= query_result.set_index("Date")
        cls.test_data  = {
            ticker: query_result[query_result["Ticker"]==ticker]["price_close"] for ticker in cls.tickers
        }


        



    @classmethod
    def extract_features(cls):
        pass

        


    @classmethod
    def _execute_train(cls): 

        cls.env = TradingEnv(cls.train_data, window = cls.window)
        check_env(cls.env, warn=True)

        cls.model = PPO("MlpPolicy", cls.env,
                    verbose=1,
                    device="cuda",
                        ent_coef=0.01, n_steps=4096, batch_size=256)
        #cls.model.learn(total_timesteps=1_000_000)
        cls.model.learn(total_timesteps=10_000)



    @classmethod 
    def initialize_architecture(cls): 
        pass