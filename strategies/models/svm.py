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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
from typing import Optional


class SVMStrategy(Strategy): 

    strategy_name = "SVM" 
    strategy_file_name = strategy_name + ".json"
    explanation = "Support Vector Machine classifier strategy using technical indicators" 

    feature_cols = [
        "SMA_20", 
        "SMA_50", 
        "Volatility_20", 
        "returns", 
        "RSI_14", 
        "MACD", 
        "MACD_Signal", 
        "MACD_Hist"
    ]
    target_col = "Target"

    @classmethod
    def get_model_extension(cls) -> str:
        return ".pkl"

    @classmethod
    def _save_model_to_disk(cls, path: str):
        # Save the scikit-learn SVC model to path
        joblib.dump(cls.model, path)
        dir_path = os.path.dirname(path)
        joblib.dump(cls.scaler, os.path.join(dir_path, "feature_scaler.pkl"))

        hyperparams_path = os.path.join(dir_path, "hyperparams.json")
        with open(hyperparams_path, "w") as f:
            json.dump(cls.hyperparams, f, indent=4)

    @classmethod
    def _load_model_from_disk(cls, path: str):
        cls.model = joblib.load(path)
        dir_path = os.path.dirname(path)
        cls.scaler = joblib.load(os.path.join(dir_path, "feature_scaler.pkl"))
        hyperparams_path = os.path.join(dir_path, "hyperparams.json")
        with open(hyperparams_path, "r") as f:
            cls.hyperparams = json.load(f)

    @classmethod
    def get_training_data(cls):
        """retrieve the data for training"""
        cls.data = Market.get_historical_data(
            tickers=['AAPL']
        )
        cls.data["date"] = pd.to_datetime(cls.data["date"])
        cls.data = cls.data.set_index("date")
        
        close = cls.data["price_close"]
        
        # Features calculation
        cls.sma_20 = Features.get_moving_average(close, 20)
        cls.sma_50 = Features.get_moving_average(close, 50)
        
        cls.log_returns = Metrics.get_daily_log_returns(close)
        cls.rolling_volatility = cls.log_returns.rolling(window=20).std()
        
        cls.returns = Metrics.get_daily_returns(close)
        
        # RSI 14
        cls.rsi_14 = Features.get_relative_strength_index(close, window=14)
        
        # MACD
        macd_df = Features.get_ma_convergence_divergence(close)
        cls.macd = macd_df["MACD"]
        cls.macd_signal = macd_df["MACD_signal"]
        cls.macd_hist = macd_df["MACD_hist"]
        
        # Target: 1 if next close > current close, else 0
        target = (close.shift(-1) > close).astype(object)
        target.iloc[-1] = np.nan
        
        cls.df = pd.DataFrame({
            "Close": close,
            "SMA_20": cls.sma_20,
            "SMA_50": cls.sma_50,
            "Volatility_20": cls.rolling_volatility,
            "returns": cls.returns,
            "RSI_14": cls.rsi_14,
            "MACD": cls.macd,
            "MACD_Signal": cls.macd_signal,
            "MACD_Hist": cls.macd_hist,
            "Target": target
        })
        cls.df.dropna(inplace=True)
        cls.df["Target"] = cls.df["Target"].astype(int)

    @classmethod
    def extract_features(cls):
        """implement feature extraction for the model; should be used for inference (=trading) and training"""
        cls.scaler = StandardScaler()
        scaled_data = cls.scaler.fit_transform(cls.df[cls.feature_cols])
        
        X = scaled_data
        y = cls.df[cls.target_col].values
        
        cls.train_size = int(len(X) * 0.8)
        
        cls.X_train = X[:cls.train_size]
        cls.X_test = X[cls.train_size:]
        
        cls.y_train = y[:cls.train_size]
        cls.y_test = y[cls.train_size:]

    @classmethod
    def _execute_train(cls, model_version: Optional[str] = None) -> dict:
        """train the model using SVC classifier"""
        C = cls.hyperparams.get("C", 1.0)
        kernel = cls.hyperparams.get("kernel", "rbf")
        gamma = cls.hyperparams.get("gamma", "auto")
        
        cls.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
        cls.model.fit(cls.X_train, cls.y_train)
        
        y_pred = cls.model.predict(cls.X_train)
        train_acc = accuracy_score(cls.y_train, y_pred)
        
        print(f"[{cls.strategy_name}] Train accuracy: {train_acc:.4f}")
        return {"train_accuracy": float(train_acc)}

    @classmethod
    def _execute_test(cls, model_version: Optional[str] = None) -> dict:
        """test the trained SVC strategy with confidence thresholding"""
        conf_threshold = cls.hyperparams.get("confidence_threshold", 0.50)
        print(f"[{cls.strategy_name}] Executing backtest using SVM predictions (confidence_threshold={conf_threshold})...")
        
        if hasattr(cls.model, "predict_proba"):
            probs = cls.model.predict_proba(cls.X_test)[:, 1]
            y_pred = (probs >= conf_threshold).astype(int)
        else:
            y_pred = cls.model.predict(cls.X_test)
            
        actuals = cls.y_test
        
        test_start_idx = cls.train_size
        strategy_df = cls.df.iloc[test_start_idx:].copy().reset_index()
        
        if len(strategy_df) != len(y_pred):
            strategy_df = strategy_df.iloc[:len(y_pred)]
            
        strategy_df["predicted_direction"] = y_pred
        strategy_df["actual_direction"] = actuals
        
        # Position is predicted direction from yesterday
        strategy_df["position"] = strategy_df["predicted_direction"].shift(1).fillna(0)
        
        strategy_df["market_return"] = strategy_df["returns"]
        strategy_df["strategy_return"] = strategy_df["position"] * strategy_df["market_return"]
        
        strategy_df["cumulative_market"] = (1 + strategy_df["market_return"]).cumprod()
        strategy_df["cumulative_strategy"] = (1 + strategy_df["strategy_return"]).cumprod()
        
        total_return_strategy = strategy_df["cumulative_strategy"].iloc[-1] - 1
        total_return_market = strategy_df["cumulative_market"].iloc[-1] - 1
        
        sharpe = (
            strategy_df["strategy_return"].mean() /
            (strategy_df["strategy_return"].std() + 1e-9)
        ) * np.sqrt(252)
        
        cum = strategy_df["cumulative_strategy"]
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / (rolling_max + 1e-9)
        max_drawdown = drawdown.min()
        
        accuracy = accuracy_score(actuals, y_pred)
        win_rate = (strategy_df[strategy_df["position"] == 1]["market_return"] > 0).mean() if (strategy_df["position"] == 1).sum() > 0 else 0.0
        
        print(f"Strategy Total Return : {total_return_strategy:.2%}")
        print(f"Market   Total Return : {total_return_market:.2%}")
        print(f"Sharpe Ratio          : {sharpe:.4f}")
        print(f"Max Drawdown          : {max_drawdown:.2%}")
        print(f"Prediction Accuracy   : {accuracy:.2%}")
        print(f"Trade Win Rate        : {win_rate:.2%} (Trades: {(strategy_df['position'] == 1).sum()})")
        
        return {
            "total_return_strategy": float(total_return_strategy),
            "total_return_market": float(total_return_market),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "accuracy": float(accuracy),
            "win_rate": float(win_rate),
            "num_trades": int((strategy_df['position'] == 1).sum())
        }

    @classmethod
    def get_data_for_trade(cls, ticker: str = 'AAPL', target_date=None) -> pd.DataFrame:
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
        
        close = raw_data["price_close"]
        
        # Calculate the features exactly as done in get_training_data
        sma_20 = Features.get_moving_average(close, 20)
        sma_50 = Features.get_moving_average(close, 50)
        
        log_returns = Metrics.get_daily_log_returns(close)
        rolling_vol = log_returns.rolling(window=20).std()
        
        daily_returns = Metrics.get_daily_returns(close)
        
        # RSI 14
        rsi_14 = Features.get_relative_strength_index(close, window=14)
        
        # MACD
        macd_df = Features.get_ma_convergence_divergence(close)
        
        trade_df = pd.DataFrame({
            "Close": close,
            "SMA_20": sma_20,
            "SMA_50": sma_50,
            "Volatility_20": rolling_vol,
            "returns": daily_returns,
            "RSI_14": rsi_14,
            "MACD": macd_df["MACD"],
            "MACD_Signal": macd_df["MACD_signal"],
            "MACD_Hist": macd_df["MACD_hist"]
        })
        
        trade_df.dropna(inplace=True)
        return trade_df

    @classmethod
    def extract_features_for_trade(cls, trade_df: pd.DataFrame) -> np.ndarray:
        recent_features = trade_df[cls.feature_cols].tail(1)
        scaled_data = cls.scaler.transform(recent_features)
        return scaled_data

    @classmethod
    def _execute_trade(cls, ticker: str = 'AAPL', target_date=None, model_version: Optional[str] = None) -> int:
        """
        End-to-end execution of a single trade decision using the live SVM model.
        Returns 1 for Long, 0 for Flat.
        """
        trade_df = cls.get_data_for_trade(ticker, target_date=target_date)
        X_input = cls.extract_features_for_trade(trade_df)
        
        conf_threshold = cls.hyperparams.get("confidence_threshold", 0.50)
        if hasattr(cls.model, "predict_proba"):
            prob = cls.model.predict_proba(X_input)[0, 1]
            prediction = 1 if prob >= conf_threshold else 0
        else:
            prediction = cls.model.predict(X_input)[0]
        
        print(f"[{cls.strategy_name}] Signal for {ticker} on {target_date}: {prediction}")
        
        if prediction == 1:
            cls.tradelog.append_trade(
                type=TransactionType.BUY, 
                currency=Currency.USD, 
                date=str(target_date), 
                shares=1.0, 
                security=Security(
                    name="Apple Inc.", 
                    ticker=ticker,
                    currency=Currency.USD
                )
            )
            return 1  # Long (buy)
        else:
            cls.tradelog.append_trade(
                type=TransactionType.SELL,
                currency=Currency.USD, 
                date=str(target_date), 
                shares=1.0, 
                security=Security(
                    name="Apple Inc.", 
                    ticker=ticker,
                    currency=Currency.USD
                )
            )
            return 0  # Flat (sell/do nothing)
