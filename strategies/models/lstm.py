from ..strategy import Strategy
from ..interfaces import Security, Currency, TransactionType



from portfolio.models.features import Features
from portfolio.models.metrics import Metrics 
from portfolio.models.market import Market


import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date
from typing import Optional



class LSTMStrategy(Strategy): 

    strategy_name = "LSTM" 
    strategy_file_name = strategy_name + ".json"
    explanation = "some LSTM strategy" 
    sequence_length = 60

    feature_cols = ["Close", "SMA_20", "SMA_50", "Volatility_20", "returns"]
    target_col   = "Close"



    @classmethod
    def get_model_extension(cls) -> str:
        return ".keras"

    @classmethod
    def _save_model_to_disk(cls, path: str):
        cls.model.save(path)
        dir_path = os.path.dirname(path)
        joblib.dump(cls.scaler, os.path.join(dir_path, "feature_scaler.pkl"))
        joblib.dump(cls.target_scaler, os.path.join(dir_path, "target_scaler.pkl"))


        hyperparams_path = os.path.join(dir_path, "hyperparams.json")
        with open(hyperparams_path, "w") as f:
            json.dump(cls.hyperparams, f, indent=4)

    @classmethod
    def _load_model_from_disk(cls, path: str):
        cls.model = load_model(path)
        dir_path = os.path.dirname(path)
        cls.scaler = joblib.load(os.path.join(dir_path, "feature_scaler.pkl"))
        cls.target_scaler = joblib.load(os.path.join(dir_path, "target_scaler.pkl"))
        hyperparams_path = os.path.join(dir_path, "hyperparams.json")
        with open(hyperparams_path, "r") as f:
                cls.hyperparams = json.load(f)

    @classmethod
    def get_data_for_trade(cls, ticker: str = 'AAPL', target_date=None) -> pd.DataFrame:
        from datetime import datetime, timedelta
        
        # Default to now if no date is provided
        end_date = target_date if target_date else datetime.now()
        start_date = end_date - timedelta(days=200) 
        
        raw_data = Market.get_historical_data(
            tickers=[ticker], 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d')
        )

        raw_data["date"] = pd.to_datetime(raw_data["date"])
        raw_data = raw_data.set_index("date")
        
        # Calculate the features exactly as done in get_training_data
        sma_20 = Features.get_moving_average(raw_data["price_close"], 20)
        sma_50 = Features.get_moving_average(raw_data["price_close"], 50)
        
        log_returns = Metrics.get_daily_log_returns(raw_data["price_close"])
        rolling_vol = log_returns.rolling(window=20).std()
        
        daily_returns = Metrics.get_daily_returns(raw_data["price_close"])
        
        # Combine into DataFrame, keeping only the columns needed for the model
        trade_df = pd.DataFrame({
            "Close": raw_data["price_close"],
            "SMA_20": sma_20,
            "SMA_50": sma_50,
            "Volatility_20": rolling_vol,
            "returns": daily_returns
        })
        
        # Drop the initial rows that contain NaNs from the rolling calculations
        trade_df.dropna(inplace=True)
        
        if len(trade_df) < cls.sequence_length:
            raise ValueError(f"Insufficient data. Need at least {cls.sequence_length} valid rows, but got {len(trade_df)}.")
            
        return trade_df

    
    @classmethod
    def extract_features_for_trade(cls, trade_df: pd.DataFrame) -> np.ndarray:
        """
        Scales and reshapes the most recent data into the 3D tensor format 
        required by the LSTM model for a single prediction.
        """
        recent_features = trade_df[cls.feature_cols].tail(cls.sequence_length)
        
        scaled_data = cls.scaler.transform(recent_features)
        
        X_input = np.array([scaled_data])
        
        return X_input



    @classmethod
    def _execute_trade(cls,  ticker: str = 'AAPL', target_date=None, model_version: Optional[str] = None,) -> int:
        """
        End-to-end execution of a single trade decision using the live LSTM model.
        Returns 1 for Long, 0 for Flat.
        """


        trade_df = cls.get_data_for_trade(ticker, target_date=target_date)
        
        X_input = cls.extract_features_for_trade(trade_df)
        
        scaled_prediction = cls.model.predict(X_input)
        predicted_close = cls.target_scaler.inverse_transform(scaled_prediction)[0][0]
        
        current_close = trade_df["Close"].iloc[-1]
        print(target_date)
        
        if predicted_close > current_close:
            cls.tradelog.append_trade(
                type = TransactionType.BUY, 
                currency = Currency.USD, 
                date = str(target_date), 
                shares =1.0, 
                security = Security(
                    name = "Apple Inc.", 
                    ticker= "AAPL",
                    currency = Currency.USD
                )
            )

            return 1  # Long (buy)
        else:
            cls.tradelog.append_trade(
                type = TransactionType.SELL,
                currency=   Currency.USD, 
                date = str(target_date), 
                shares =1.0, 
                security =Security(
                    name = "Apple Inc.", 
                    ticker= "AAPL",
                    currency = Currency.USD
                )
            )
            return 0  # Flat (ie sell/ do nothing)
        




    @classmethod
    def _execute_test(cls,  model_version: Optional[str] = None ) -> dict: 
        """
        Trades are strictly based on the LSTM's predicted close vs the previous day's actual clos. 
        """
        print(f"[{cls.strategy_name}] Executing backtest using LSTM predictions...")

        scaled_predictions = cls.model.predict(cls.X_test)

        predictions = cls.target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
        actuals = cls.target_scaler.inverse_transform(cls.y_test.reshape(-1, 1)).flatten()

        test_start_idx = cls.train_size + cls.sequence_length

        strategy_df = cls.df.iloc[test_start_idx:].copy().reset_index()
        
        if len(strategy_df) != len(predictions):
            strategy_df = strategy_df.iloc[:len(predictions)]

        strategy_df["predicted_close"] = predictions
        strategy_df["actual_close"] = actuals

        # decision logic 
        # to decide our position for day 't', we compare the prediction 
        # for day t vs the close from day 't-1' 
        strategy_df["previous_actual_close"] = strategy_df["actual_close"].shift(1)
        
        # padding using the last day of the training set (for the oldest date)
        strategy_df["previous_actual_close"] = strategy_df["previous_actual_close"].fillna(
            cls.df["Close"].iloc[test_start_idx - 1]
        )

        # 1 (Long) if predicted close > previous actual close; 0 (Flat) otherwise
        strategy_df["position"] = np.where(
            strategy_df["predicted_close"] > strategy_df["previous_actual_close"], 1, 0
        )
        
        strategy_df["market_return"] = strategy_df["actual_close"].pct_change().fillna(0)

        strategy_df["strategy_return"] = strategy_df["position"] * strategy_df["market_return"]

        strategy_df["cumulative_market"] = (1 + strategy_df["market_return"]).cumprod()
        strategy_df["cumulative_strategy"] = (1 + strategy_df["strategy_return"]).cumprod()

        total_return_strategy = strategy_df["cumulative_strategy"].iloc[-1] - 1
        total_return_market   = strategy_df["cumulative_market"].iloc[-1] - 1

        sharpe = (
            strategy_df["strategy_return"].mean() /
            (strategy_df["strategy_return"].std() + 1e-9)
        ) * np.sqrt(252)

        cum = strategy_df["cumulative_strategy"]
        rolling_max  = cum.cummax()
        drawdown     = (cum - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        print(f"Strategy Total Return : {total_return_strategy:.2%}")
        print(f"Market   Total Return : {total_return_market:.2%}")
        print(f"Sharpe Ratio          : {sharpe:.4f}")
        print(f"Max Drawdown          : {max_drawdown:.2%}")

        return {
            "total_return_strategy": float(total_return_strategy),
            "total_return_market": float(total_return_market),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown)
        }







    @classmethod
    def get_training_data(cls):
        """retrieve the data for training"""

        cls.data = Market.get_historical_data(
            tickers = ['AAPL'], 
            #start = start, 
            #end  = end, 
        )
        cls.data["date"] = pd.to_datetime(cls.data["date"])
        cls.data = cls.data.set_index("date")
        cls.sma_20 = Features.get_moving_average(
            cls.data["price_close"], 
            20 
        )
        cls.sma_50 = Features.get_moving_average(
            cls.data["price_close"], 
            50 
        )
        cls.sma_200 = Features.get_moving_average(
            cls.data["price_close"], 
            200 
        )
        cls.sma_50.dropna(inplace = True)
        cls.sma_20.dropna(inplace = True)
        cls.sma_200.dropna(inplace = True)


        cls.log_returns= Metrics.get_daily_log_returns(cls.data["price_close"])
        cls.log_returns.dropna(inplace = True)

        cls.rolling_volatility = cls.log_returns.rolling(window  = 20).std()
        cls.rolling_volatility.dropna(inplace = True)
        cls.returns = Metrics.get_daily_returns(cls.data["price_close"])
        cls.returns.dropna(inplace = True)

        cls.df = pd.DataFrame({
            "Close": cls.data["price_close"], 
            "Volume": cls.data["volume"], 
            "SMA_20": cls.sma_20, 
            "SMA_50": cls.sma_50, 
            "SMA_200": cls.sma_200, 
            "Log_return": cls.log_returns, 
            "Volatility_20": cls.rolling_volatility, 
            "returns": cls.returns,
        }
        )
        cls.df.dropna(inplace = True)
        #cls.df.sort_index(inplace = True)
        cls.df["SMA_Signal"] = np.where(cls.df["SMA_20"] > cls.df["SMA_50"], 1, -1)

        



    @classmethod
    def extract_features(cls):
        """implement feature extraction for the model; should be used for inference (=trading) and training"""
        cls.scaler = MinMaxScaler(feature_range = (0,1))

        scaled_data = cls.scaler.fit_transform(cls.df[cls.feature_cols])


        cls.target_scaler = MinMaxScaler(feature_range=(0, 1))
        cls.target_scaled = cls.target_scaler.fit_transform(cls.df[[cls.target_col]])



        X, y = [], []

        for i in range(cls.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-cls.sequence_length:i, :])
            y.append(cls.target_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        #X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        cls.train_size = int(len(X)*0.8)


        cls.X_train = X[:cls.train_size]
        cls.X_test = X[cls.train_size:]

        cls.y_train = y[:cls.train_size]
        cls.y_test = y[cls.train_size:]


        


    @classmethod
    def _execute_train(cls, model_version: Optional[str] = None): 
        """
        
        train the model. this method is supposed to be implemented for production training; not development

        get the data, extract features, train the model 
        
        """



        cls.initialize_architecture()
        history = cls.model.fit(
            cls.X_train,
            cls.y_train,
            epochs=10,
            batch_size=32,
            validation_data=(cls.X_test, cls.y_test),
            verbose=1
        )
        return history.history






    @classmethod 
    def initialize_architecture(cls): 

        

        cls.model = Sequential()

        
        cls.model.add(
            LSTM(
                units=cls.hyperparams["lstm_units"], 
                return_sequences=True, 
                input_shape=(cls.X_train.shape[1], cls.X_train.shape[2])
                )
        )


        cls.model.add(
            Dropout(cls.hyperparams["dropout"])
        )

        cls.model.add(LSTM(units=cls.hyperparams["lstm_units"]))

        cls.model.add(Dropout(cls.hyperparams["dropout"]))

        cls.model.add(Dense(cls.hyperparams["dense_output_layers"]))

        cls.model.compile(optimizer="adam", loss="mean_squared_error")
