from ..strategy import Strategy



import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt



CLF_BASE = 'clf'
REG_BASE = 'reg'

CLF_SCORING = 'f1'
REG_SCORING = 'r2'


class RandomForestStrategy(Strategy): 
    strategy_name = "Random Rorests"

    data = None



    n_iter = 30
    base = CLF_SCORING
    scoring = CLF_SCORING 
    verbose = 1

    param_grid = {
            'n_estimators' : [100, 200, 300, 500],
            'max_depth' : [5, 10, 20, None],
            'max_features' : ['sqrt', 'log2', 0.5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4]
    }


    tscv = TimeSeriesSplit(n_splits=5)
    clf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    reg_base = RandomForestRegressor(random_state=42)



    @classmethod
    def train(cls): 
        cls.train_classification()
        cls.train_regression()
        return


    @classmethod 
    def train_regression(cls): 
    

        estimator = cls.reg_base

        search = RandomizedSearchCV(
            estimator = estimator,
            param_distributions= cls.param_grid,
            n_iter = cls.n_iter,
            scoring = cls.scoring,
            cv = cls.tscv,
            n_jobs = -1,
            random_state = 42,
            verbose = cls.verbose
        )

        search.fit(cls.X_reg_train, cls.y_reg_train)
        estimator = search.best_estimator_

        y_pred = estimator.pedict(cls.X_reg_test)

        cls.reg_mae = mean_absolute_error(cls.y_reg_test, y_pred)
        cls.reg_r2_score = r2_score(cls.y_reg_test, y_pred)
        cls.best_reg =estimator 
        return

    @classmethod 
    def train_classification(cls): 
        

        estimator = cls.clf_base
        search = RandomizedSearchCV(
            estimator = estimator,
            param_distributions= cls.param_grid,
            n_iter = cls.n_iter,
            scoring = cls.scoring,
            cv = cls.tscv,
            n_jobs = -1,
            random_state = 42,
            verbose = cls.verbose
        )

        search.fit(cls.X_clf_train, cls.y_clf_train)
        estimator = search.best_estimator_
        best_score = round(search.best_score_, 4)

        y_pred = estimator.pedict(cls.X_clf_test)
        cls.clf_report = classification_report(cls.y_clf_test, y_pred, target_names=['Down (0)', 'Up (1)'])
        cls.best_clf = estimator


    @classmethod
    def classifier(cls, X):
        return cls.best_clf.predict(X)
    

    @classmethod
    def regression(self, X, y, X_test, y_test):
        best_reg, _ = self.find_best_params(X, y, base=REG_BASE, scoring=REG_SCORING, verbose=1)
        y_pred = best_reg.predict(X_test)

        self.reg_mae = mean_absolute_error(y_test, y_pred)
        self.reg_r2_score = r2_score(y_test, y_pred)
        self.best_reg = best_reg

        print(f"MAE : {self.reg_mae:.4f}")
        print(f"R² : {self.reg_r2_score:.4f}")





    @classmethod
    def extract_features(cls): 
        #: here we extract the features
        cls.add_indicators()
        cls.add_volume_indicators()
        cls.add_time_features()
        cls.add_target()



    @classmethod
    def get_training_data(cls):
        cls.data.dropna().reset_index(drop=True)

        cls.X_clf_train, cls.y_clf_train, cls.X_clf_test, cls.y_clf_test = cls.get_train_test_clf()
        cls.X_reg_train, cls.y_reg_train, cls.X_reg_test, cls.y_reg_test = cls.get_train_test_reg()
        

    @classmethod
    def get_train_test_clf(cls):
        drop_cols = ['trend', 'Close', 'Datetime']
        X_clf = cls.data.drop(columns=[c for c in drop_cols if c in cls.data.columns])
        y_clf = cls.data['trend']

        split = int(len(X_clf) * 0.8)
        X_clf_train = X_clf.iloc[:split]
        X_clf_test = X_clf.iloc[split:]

        y_clf_train = y_clf.iloc[:split]
        y_clf_test = y_clf.iloc[split:]

        print(f"RF for Classification -> Train size: {len(X_clf_train)} | Test size: {len(X_clf_test)}")
        return X_clf_train, y_clf_train, X_clf_test, y_clf_test
    

    @classmethod
    def get_train_test_reg(cls):
        drop_cols = ['trend', 'Close', 'Datetime']
        X_reg = cls.data.drop(columns=[c for c in drop_cols if c in cls.data.columns])
        y_reg = cls.data['Close']

        split = int(len(X_reg) * 0.8)
        X_reg_train = X_reg.iloc[:split]
        X_reg_test = X_reg.iloc[split:]

        y_reg_train = y_reg.iloc[:split]
        y_reg_test = y_reg.iloc[split:]

        print(f"RF for Regression -> Train size: {len(X_reg_train)} | Test size: {len(X_reg_test)}")
        return X_reg_train, y_reg_train, X_reg_test, y_reg_test



    
    @classmethod
    def add_indicators(cls):
        close = cls.data['Close']

        # SMA (5, 20, 50)
        for period in [5, 20, 50]:
            cls.data[f"sma_{period}"] = close.rolling(window=period).mean()

        # EMA (5, 10, 20, 50)
        for period in [5, 10, 20, 50]:
            cls.data[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()
        
        # MACD(12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        cls.data["macd"] = ema_12 - ema_26
        cls.data["macd_signal"] = cls.data["macd"].ewm(span=9, adjust=False).mean()
        cls.data["macd_hist"] = cls.data["macd"] - cls.data["macd_signal"]

        # RSI
        cls.data["rsi_14"] = cls.compute_rsi(close, 14)
        cls.data["rsi_7"] = cls.compute_rsi(close, 7)


    @classmethod
    def add_volume_indicators(cls):
        high, low, close, volume = cls.data["High"], cls.data["Low"], cls.data["Close"], cls.data['Volume']
        prev_close = close.shift(1)

        # ATR (14)
        true_range = pd.concat([
            high - low,                  # Current high - low
            (high - prev_close).abs(),   # |High - Previous close|
            (low - prev_close).abs(),    # |Low - Previous close|
        ], axis=1).max(axis=1)

        cls.data["atr_14"] = true_range.ewm(com=13, adjust=False).mean()

        # Volume SMA (5, 20)
        for period in [5, 20]:
            cls.data[f"volume_sma_{period}"] = volume.rolling(window=period).mean()
        
        # OBV
        direction = pd.Series(0, index=cls.data.index)
        direction[close > prev_close] = 1
        direction[close < prev_close] = -1

        cls.data["obv"] = (direction * volume).cumsum()


    @classmethod
    def add_time_features(cls):
        close = cls.data['Close']

        # Log Returns (1, 3, 5, 20)
        for period in [1, 3, 5, 20]:
            cls.data[f"return_{period}"] = (close / close.shift(period)).apply(lambda x: math.log(x))
        
        # Lagged Returns
        ret_1 = cls.data["return_1"]
        for lag in [1, 3, 5, 20]:
            cls.data[f"lagged_return_{lag}"] = ret_1.shift(lag)
        
        # Time Features
        t = pd.to_datetime(cls.data["Datetime"])
        cls.data["hour_of_day"] = t.dt.hour 
        cls.data["minute_of_hour"] = t.dt.minute 
        cls.data["day_of_week"] = t.dt.dayofweek + 1 

    

    @classmethod
    def add_target(cls):
        cls.data['trend'] = (cls.data["Close"] > cls.data["Close"].shift(1)).astype(int) 

    @staticmethod
    def compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        #: TODO: this function should be part of the portfolio repo
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean() # Wilder smoothing
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    
    @classmethod
    def extract_features(cls): 
        return 


    