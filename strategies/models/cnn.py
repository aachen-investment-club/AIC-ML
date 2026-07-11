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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date
from typing import Optional


class CNNStrategy(Strategy):
    """
    1D CNN strategy for predicting next-day returns.
    
    The CNN extracts temporal patterns from rolling windows of market data
    and predicts the direction/magnitude of next-day returns.
    """

    strategy_name = "CNN"
    strategy_file_name = strategy_name + ".json"
    explanation = "1D CNN strategy predicting next-day returns"
    sequence_length = 60

    feature_cols = ["Close", "SMA_20", "SMA_50", "Volatility_20", "returns"]
    target_col = "future_return"  # Predict next-day return, not price

    # Default hyperparameters
    hyperparams = {
        "conv_filters": 64,
        "kernel_size": 3,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "patience": 5,
    }

    @classmethod
    def get_model_extension(cls) -> str:
        return ".keras"

    @classmethod
    def _save_model_to_disk(cls, path: str):
        """Save model, scalers, and hyperparameters to disk."""
        cls.model.save(path)
        dir_path = os.path.dirname(path)
        joblib.dump(cls.scaler, os.path.join(dir_path, "feature_scaler.pkl"))
        joblib.dump(cls.target_scaler, os.path.join(dir_path, "target_scaler.pkl"))

        hyperparams_path = os.path.join(dir_path, "hyperparams.json")
        with open(hyperparams_path, "w") as f:
            json.dump(cls.hyperparams, f, indent=4)

    @classmethod
    def _load_model_from_disk(cls, path: str):
        """Load model, scalers, and hyperparameters from disk."""
        cls.model = load_model(path)
        dir_path = os.path.dirname(path)
        cls.scaler = joblib.load(os.path.join(dir_path, "feature_scaler.pkl"))
        cls.target_scaler = joblib.load(os.path.join(dir_path, "target_scaler.pkl"))

        hyperparams_path = os.path.join(dir_path, "hyperparams.json")
        with open(hyperparams_path, "r") as f:
            cls.hyperparams = json.load(f)

    @classmethod
    def get_data_for_trade(cls, ticker: str = "AAPL", target_date=None) -> pd.DataFrame:
        """
        Fetch recent market data for making a live trade decision.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Date for the trade decision (defaults to now)
            
        Returns:
            DataFrame with features ready for prediction
        """
        from datetime import datetime, timedelta

        end_date = target_date if target_date else datetime.now()
        start_date = end_date - timedelta(days=200)

        raw_data = Market.get_historical_data(
            tickers=[ticker],
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )

        raw_data["date"] = pd.to_datetime(raw_data["date"])
        raw_data = raw_data.set_index("date")

        # Calculate features exactly as done in training
        sma_20 = Features.get_moving_average(raw_data["price_close"], 20)
        sma_50 = Features.get_moving_average(raw_data["price_close"], 50)

        log_returns = Metrics.get_daily_log_returns(raw_data["price_close"])
        rolling_vol = log_returns.rolling(window=20).std()

        daily_returns = Metrics.get_daily_returns(raw_data["price_close"])

        trade_df = pd.DataFrame({
            "Close": raw_data["price_close"],
            "SMA_20": sma_20,
            "SMA_50": sma_50,
            "Volatility_20": rolling_vol,
            "returns": daily_returns,
        })

        trade_df.dropna(inplace=True)

        if len(trade_df) < cls.sequence_length:
            raise ValueError(
                f"Insufficient data. Need at least {cls.sequence_length} valid rows, "
                f"but got {len(trade_df)}."
            )

        return trade_df

    @classmethod
    def extract_features_for_trade(cls, trade_df: pd.DataFrame) -> np.ndarray:
        """
        Scale and reshape recent data into the 3D tensor format required by the CNN.
        
        Args:
            trade_df: DataFrame with feature columns
            
        Returns:
            3D array of shape (1, sequence_length, n_features)
        """
        recent_features = trade_df[cls.feature_cols].tail(cls.sequence_length)
        scaled_data = cls.scaler.transform(recent_features)
        X_input = np.array([scaled_data])
        return X_input

    @classmethod
    def _execute_trade(
        cls, ticker: str = "AAPL", target_date=None, model_version: Optional[str] = None
    ) -> int:
        """
        Execute a single trade decision using the CNN model.
        
        The CNN predicts next-day return. If predicted return > 0, go Long.
        Otherwise, go Flat (sell/do nothing).
        
        Returns:
            1 for Long (buy), 0 for Flat (sell/hold)
        """
        trade_df = cls.get_data_for_trade(ticker, target_date=target_date)
        X_input = cls.extract_features_for_trade(trade_df)

        # Predict next-day return
        scaled_prediction = cls.model.predict(X_input, verbose=0)
        predicted_return = cls.target_scaler.inverse_transform(scaled_prediction)[0][0]

        print(f"[{cls.strategy_name}] Target date: {target_date}")
        print(f"[{cls.strategy_name}] Predicted next-day return: {predicted_return:.4%}")

        if predicted_return > 0:
            cls.tradelog.append_trade(
                type=TransactionType.BUY,
                currency=Currency.USD,
                date=str(target_date),
                shares=1.0,
                security=Security(
                    name="Apple Inc.",
                    ticker=ticker,
                    currency=Currency.USD,
                ),
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
                    currency=Currency.USD,
                ),
            )
            return 0  # Flat (sell/do nothing)

    @classmethod
    def _execute_test(cls, model_version: Optional[str] = None) -> dict:
        """
        Backtest the CNN strategy on the test set.
        
        Trades are based on predicted return direction:
        - Predicted return > 0 → Long
        - Predicted return <= 0 → Flat
        
        Returns:
            Dictionary with backtest metrics
        """
        print(f"[{cls.strategy_name}] Executing backtest using CNN predictions...")

        # Get predictions on test set
        scaled_predictions = cls.model.predict(cls.X_test, verbose=0)
        pred_returns = cls.target_scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).flatten()
        actual_returns = cls.target_scaler.inverse_transform(
            cls.y_test.reshape(-1, 1)
        ).flatten()

        # Build strategy DataFrame
        test_start_idx = cls.train_size + cls.sequence_length
        strategy_df = cls.df.iloc[test_start_idx:].copy().reset_index()

        if len(strategy_df) != len(pred_returns):
            strategy_df = strategy_df.iloc[: len(pred_returns)]

        strategy_df["predicted_return"] = pred_returns
        strategy_df["actual_return"] = actual_returns

        # Position: Long (1) if predicted return > 0, Flat (0) otherwise
        strategy_df["signal"] = np.where(strategy_df["predicted_return"] > 0, 1, 0)
        # Shift position by 1 to avoid look-ahead bias
        strategy_df["position"] = strategy_df["signal"].shift(1).fillna(0)

        # Calculate returns
        strategy_df["market_return"] = strategy_df["Close"].pct_change().fillna(0)
        strategy_df["strategy_return"] = strategy_df["position"] * strategy_df["market_return"]

        # Cumulative returns
        strategy_df["cumulative_market"] = (1 + strategy_df["market_return"]).cumprod()
        strategy_df["cumulative_strategy"] = (1 + strategy_df["strategy_return"]).cumprod()

        # Performance metrics
        total_return_strategy = strategy_df["cumulative_strategy"].iloc[-1] - 1
        total_return_market = strategy_df["cumulative_market"].iloc[-1] - 1

        sharpe = (
            strategy_df["strategy_return"].mean()
            / (strategy_df["strategy_return"].std() + 1e-9)
        ) * np.sqrt(252)

        cum = strategy_df["cumulative_strategy"]
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Directional accuracy
        direction_acc = np.mean(np.sign(pred_returns) == np.sign(actual_returns))

        print(f"Strategy Total Return  : {total_return_strategy:.2%}")
        print(f"Market Total Return    : {total_return_market:.2%}")
        print(f"Sharpe Ratio           : {sharpe:.4f}")
        print(f"Max Drawdown           : {max_drawdown:.2%}")
        print(f"Directional Accuracy   : {direction_acc:.2%}")

        return {
            "total_return_strategy": float(total_return_strategy),
            "total_return_market": float(total_return_market),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "directional_accuracy": float(direction_acc),
        }

    @classmethod
    def get_training_data(cls):
        """Retrieve and prepare training data from market API."""
        cls.data = Market.get_historical_data(tickers=["AAPL"])
        cls.data["date"] = pd.to_datetime(cls.data["date"])
        cls.data = cls.data.set_index("date")

        # Calculate features
        cls.sma_20 = Features.get_moving_average(cls.data["price_close"], 20)
        cls.sma_50 = Features.get_moving_average(cls.data["price_close"], 50)

        cls.log_returns = Metrics.get_daily_log_returns(cls.data["price_close"])
        cls.rolling_volatility = cls.log_returns.rolling(window=20).std()
        cls.returns = Metrics.get_daily_returns(cls.data["price_close"])

        # Target: next-day return
        cls.future_return = cls.returns.shift(-1)

        cls.df = pd.DataFrame({
            "Close": cls.data["price_close"],
            "SMA_20": cls.sma_20,
            "SMA_50": cls.sma_50,
            "Volatility_20": cls.rolling_volatility,
            "returns": cls.returns,
            "future_return": cls.future_return,
        })
        cls.df.dropna(inplace=True)

    @classmethod
    def extract_features(cls):
        """
        Extract and scale features, build sequences for CNN training.
        
        Creates train/test split and stores X_train, X_test, y_train, y_test.
        """
        cls.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = cls.scaler.fit_transform(cls.df[cls.feature_cols])

        cls.target_scaler = MinMaxScaler(feature_range=(0, 1))
        cls.target_scaled = cls.target_scaler.fit_transform(cls.df[[cls.target_col]])

        # Build sequences
        X, y = [], []
        for i in range(cls.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - cls.sequence_length : i, :])
            y.append(cls.target_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Train/test split
        cls.train_size = int(len(X) * 0.8)

        cls.X_train = X[: cls.train_size]
        cls.X_test = X[cls.train_size :]
        cls.y_train = y[: cls.train_size]
        cls.y_test = y[cls.train_size :]

    @classmethod
    def initialize_architecture(cls):
        """
        Build the 1D CNN model architecture.
        
        Architecture:
        - Input: (sequence_length, n_features)
        - Conv1D (64 filters, kernel_size=3, causal padding) + BatchNorm + Dropout
        - Conv1D (64 filters, kernel_size=3, causal padding) + BatchNorm + Dropout
        - GlobalAveragePooling1D
        - Dense(1) output for return prediction
        """
        n_features = len(cls.feature_cols)

        cnn_input = Input(shape=(cls.sequence_length, n_features))

        x = Conv1D(
            cls.hyperparams["conv_filters"],
            kernel_size=cls.hyperparams["kernel_size"],
            activation="relu",
            padding="causal",
        )(cnn_input)
        x = BatchNormalization()(x)
        x = Dropout(cls.hyperparams["dropout"])(x)

        x = Conv1D(
            cls.hyperparams["conv_filters"],
            kernel_size=cls.hyperparams["kernel_size"],
            activation="relu",
            padding="causal",
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(cls.hyperparams["dropout"])(x)

        x = GlobalAveragePooling1D(name="cnn_features")(x)
        cnn_output = Dense(1)(x)

        cls.model = Model(inputs=cnn_input, outputs=cnn_output)
        cls.model.compile(optimizer="adam", loss="mean_squared_error")

    @classmethod
    def _execute_train(cls, model_version: Optional[str] = None) -> dict:
        """
        Train the CNN model.
        
        Uses early stopping with validation split from training data to prevent
        overfitting and data leakage.
        
        Returns:
            Training history dictionary
        """
        cls.get_training_data()
        cls.extract_features()
        cls.initialize_architecture()

        # Create validation split from training data (20% of train -> validation)
        val_size = int(len(cls.X_train) * 0.2)
        X_train_split = cls.X_train[:-val_size]
        y_train_split = cls.y_train[:-val_size]
        X_val = cls.X_train[-val_size:]
        y_val = cls.y_train[-val_size:]

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=cls.hyperparams["patience"],
            restore_best_weights=True,
        )

        history = cls.model.fit(
            X_train_split,
            y_train_split,
            epochs=cls.hyperparams["epochs"],
            batch_size=cls.hyperparams["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1,
            shuffle=False,
        )

        return history.history
