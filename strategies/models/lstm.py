from ..strategy import Strategy



from portfolio.models.features import Features
from portfolio.models.metrics import Metrics 
from portfolio.models.market import Market



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date



class LSTMStrategy(Strategy): 


    strategy_name = "LSTM_strategy" 
    strategy_file_name = strategy_name + ".json"
    explanation = "some LSTM strategy" 
    sequence_length = 60

    feature_cols = ["Close", "SMA_20", "SMA_50", "Volatility_20", "returns"]
    target_col   = "Close"



    @classmethod
    def trade(cls): 
        """
        - when called, this should fetch the data, extract features and perform trades
        - new trades should added to the tradelog 
        """
        # Apply simple SMA crossover idea, can be improvised later

        # align predictions with the original df index

        predictions = cls.model.predict(cls.X_test)

        # Inverse transform
        predictions = cls.target_scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = cls.target_scaler.inverse_transform(cls.y_test.reshape(-1, 1))


        test_start_idx = cls.train_size + cls.sequence_length   # first index in df corresponding to test set

        strategy_df = cls.df.iloc[test_start_idx:].copy().reset_index()
        strategy_df["predicted_close"] = predictions.flatten()
        strategy_df["actual_close"]    = y_test_actual.flatten()


        # Signal, +1 (Long) when SMA_20 > SMA_50; -1 (Short/Flat) otherwise
        strategy_df["position"] = np.where(
            strategy_df["SMA_20"] > strategy_df["SMA_50"], 1, 0   # 0 = flat, 1 = long
        )

        # Shift by 1 so we trade on the next day's open  & avoid look-ahead bias
        strategy_df["position"] = strategy_df["position"].shift(1).fillna(0)

        # Market daily return  or actual returns
        strategy_df["market_return"]   = strategy_df["actual_close"].pct_change().fillna(0)

        # Strategy return s
        strategy_df["strategy_return"] = strategy_df["position"] * strategy_df["market_return"]

        # cumulative returns
        strategy_df["cumulative_market"]   = (1 + strategy_df["market_return"]).cumprod()
        strategy_df["cumulative_strategy"] = (1 + strategy_df["strategy_return"]).cumprod()

        # performance metrics
        total_return_strategy = strategy_df["cumulative_strategy"].iloc[-1] - 1
        total_return_market   = strategy_df["cumulative_market"].iloc[-1] - 1

        # Annualised Sharpe Ratio 
        sharpe = (
            strategy_df["strategy_return"].mean() /
            (strategy_df["strategy_return"].std() + 1e-9)
        ) * np.sqrt(252)

        # Max Drawdown
        cum = strategy_df["cumulative_strategy"]
        rolling_max  = cum.cummax()
        drawdown     = (cum - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        print(f"Strategy Total Return : {total_return_strategy:.2%}")
        print(f"Market   Total Return : {total_return_market:.2%}")
        print(f"Sharpe Ratio          : {sharpe:.4f}")
        print(f"Max Drawdown          : {max_drawdown:.2%}")


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
    def train(cls): 
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




    @classmethod 
    def initialize_architecture(cls): 

        cls.model = Sequential()

        
        cls.model.add(LSTM(units=64, return_sequences=True, input_shape=(cls.X_train.shape[1], cls.X_train.shape[2])))


        cls.model.add(Dropout(0.2))

        cls.model.add(LSTM(units=64))
        cls.model.add(Dropout(0.2))

        cls.model.add(Dense(1))

        cls.model.compile(optimizer="adam", loss="mean_squared_error")