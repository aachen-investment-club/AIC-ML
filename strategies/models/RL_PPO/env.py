from itertools import combinations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from portfolio.models.features import Features

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # actions
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, price_data: dict, initial_cash: float = 10_000.0, window: int = 10,
                 corr_window: int = 20, min_episode_len: int = 252, randomize_start: bool = True):
        super().__init__()
        self.tickers = list(price_data.keys())
        self.n = len(self.tickers)
        self.initial_cash = initial_cash

        #: we assume that if we invest all our endowment, we have an equal distribution.
        self.slot_budget = initial_cash / self.n
        self.window = window
        self.corr_window = corr_window
        self.min_episode_len = min_episode_len
        self.randomize_start = randomize_start
        self.n_pairs = self.n * (self.n - 1) // 2  # unique off-diagonal correlations

        self._build_features(price_data)

        # n*window*4 (per-stock features) + n_pairs (pairwise correlations) + n (position flags) + 1 (cash ratio)
        obs_size = self.n * window * 4 + self.n_pairs + self.n + 1
        # considered features: RSI, returns, sma ratio, macd. 



        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * self.n)

    def _build_features(self, price_data: dict):
        all_features, all_prices, all_returns = [], [], []

        for ticker in self.tickers:
            close = pd.Series(price_data[ticker]).reset_index(drop=True).astype(float)

            feat = pd.DataFrame()
            ret = close.pct_change().fillna(0)
            feat["return"] = ret
            feat["sma_ratio"] = (close.rolling(5).mean() / close).fillna(1)

            """
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            feat["rsi"] = (100 - 100 / (1 + gain / loss)).fillna(50) / 100
            """


            feat["rsi"] = Features.get_relative_strength_index(
                close, window = 14 
            ).fillna(50)/100



            #ema12 = close.ewm(span=12, adjust=False).mean()
            #ema26 = close.ewm(span=26, adjust=False).mean()



            #feat["macd"] = ((ema12 - ema26) / close).fillna(0)
            feat["macd"] = Features.get_ma_convergence_divergence(close, 12, 26)["MACD"].fillna(0)

            all_features.append(feat.values.astype(np.float32))
            all_prices.append(close.values.astype(np.float32))
            all_returns.append(ret)

        self.feature_data = np.stack(all_features, axis=1)  # (T, n, 4)
        self.prices = np.stack(all_prices, axis=1)           # (T, n)

        # pairwise rolling correlations of returns — shape (T, n_pairs)
        returns_df = pd.concat(all_returns, axis=1)
        returns_df.columns = self.tickers
        pair_corrs = []
        for i, j in combinations(range(self.n), 2):
            corr = (returns_df.iloc[:, i]
                    .rolling(self.corr_window)
                    .corr(returns_df.iloc[:, j])
                    .fillna(0)
                    .clip(-1, 1))
            pair_corrs.append(corr.values)
        self.corr_data = np.stack(pair_corrs, axis=1).astype(np.float32)  # (T, n_pairs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        T = len(self.prices)
        if self.randomize_start:

            #: episode can start anywhere, as long as at least a single episode replayable. 
            max_start = T - self.min_episode_len
            self.t = self.np_random.integers(self.window, max(self.window + 1, max_start))
        else:
            self.t = self.window
        self.cash = float(self.initial_cash)
        self.shares = np.zeros(self.n, dtype=np.float64)
        self.prev_value = self.initial_cash
        return self._obs(), {}

    def step(self, action):
        # action is always a n_ticker - dim vector with elements in 0,1,2
        prices = self.prices[self.t]

        for i, act in enumerate(action):
            if act == self.BUY and self.cash >= prices[i] and self.shares[i] == 0: 
                #: here buy <=> max out budget for the asset. 
                spend = min(self.cash, self.slot_budget)
                self.shares[i] = spend / prices[i]
                self.cash -= spend
            elif act == self.SELL and self.shares[i] > 0:
                self.cash += self.shares[i] * prices[i]
                self.shares[i] = 0.0

        portfolio_value = self.cash + float(np.sum(self.shares * prices))

        #: reward defined as overall returns
        reward = float((portfolio_value - self.prev_value) / self.initial_cash)
        self.prev_value = portfolio_value

        #: t is the upperbound of the time window.
        self.t += 1
        terminated = bool(self.t >= len(self.prices))

        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, False, {"portfolio_value": portfolio_value}

    def _obs(self):
        window_features = self.feature_data[self.t - self.window : self.t]
        flat_features = window_features.transpose(1, 0, 2).flatten()  # (n * window * 4,)

        corr = self.corr_data[self.t]                          # (n_pairs,)
        position_flags = (self.shares > 0).astype(np.float32)  # (n,)
        cash_ratio = np.float32(self.cash / self.initial_cash)

        return np.concatenate([flat_features, corr, position_flags, [cash_ratio]]).astype(np.float32)

    def render(self):
        prices = self.prices[self.t - 1]
        value = self.cash + float(np.sum(self.shares * prices))
        positions = " | ".join(
            f"{self.tickers[i]}={'LONG' if self.shares[i] > 0 else 'CASH'}"
            for i in range(self.n)
        )
        corr = self.corr_data[self.t - 1]
        pairs = list(combinations(range(self.n), 2))
        corr_str = " | ".join(
            f"ρ({self.tickers[i]},{self.tickers[j]})={corr[k]:.2f}"
            for k, (i, j) in enumerate(pairs)
        )
        report = f"t={self.t:4d} | value={value:10.2f} | {positions} | {corr_str}"
        print(report)
        return  report
        
