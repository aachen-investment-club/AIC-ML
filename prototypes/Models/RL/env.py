from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, price_data: dict, initial_cash: float = 10_000.0, window: int = 10,
                 corr_window: int = 20, min_episode_len: int = 252, randomize_start: bool = True,
                 portfolio_state: dict | None = None, sharpe_window: int = 20):
        super().__init__()
        self.tickers = list(price_data.keys())
        self.n = len(self.tickers)
        self.initial_cash = initial_cash

        #: reward is a rolling Sharpe ratio of per-step portfolio returns,
        #: computed over the last `sharpe_window` steps.
        self.sharpe_window = sharpe_window
        self.returns_history = deque(maxlen=sharpe_window)

        #: lets the env be (re)started from an existing portfolio (e.g. live
        #: holdings) instead of always rebuilding it via simulated trades.
        #: expected shape: {"cash": float, "shares": {ticker: float, ...}}
        self.initial_portfolio_state = portfolio_state

        self.window = window
        self.corr_window = corr_window
        self.min_episode_len = min_episode_len
        self.randomize_start = randomize_start
        self.n_pairs = self.n * (self.n - 1) // 2  # unique off-diagonal correlations

        self._build_features(price_data)

        # n*window*4 (per-stock features) + n_pairs (pairwise correlations) + n (position ratios) + 1 (cash ratio)
        obs_size = self.n * window * 4 + self.n_pairs + self.n + 1
        # considered features: RSI, returns, sma ratio, macd.



        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        #: action[i] is a raw allocation score for ticker i (SB3's recommended
        #: symmetric/normalized [-1, 1] range). Together with an implicit fixed
        #: score of 0 for cash, these are softmaxed into portfolio weights that
        #: sum to 1 across all n tickers *and* cash — i.e. the agent freely
        #: redistributes the entire portfolio value (not fixed per-ticker
        #: budgets) every step, and only the (possibly fractional) difference
        #: between the current and target position is traded.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def _build_features(self, price_data: dict):
        all_features, all_prices, all_returns = [], [], []

        for ticker in self.tickers:
            close = pd.Series(price_data[ticker]).reset_index(drop=True).astype(float)

            feat = pd.DataFrame()
            ret = close.pct_change().fillna(0)
            feat["return"] = ret
            feat["sma_ratio"] = (close.rolling(5).mean() / close).fillna(1)

            delta = close.diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            feat["rsi"] = (100 - 100 / (1 + gain / loss)).fillna(50) / 100

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            feat["macd"] = ((ema12 - ema26) / close).fillna(0)

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

        #: an explicit `options["portfolio_state"]` overrides the one passed to
        #: __init__, which in turn overrides the default fresh-cash start.
        portfolio_state = (options or {}).get("portfolio_state", self.initial_portfolio_state)
        if portfolio_state is not None:
            self.cash = float(portfolio_state.get("cash", self.initial_cash))
            shares = portfolio_state.get("shares", {})
            self.shares = np.array([float(shares.get(t, 0.0)) for t in self.tickers], dtype=np.float64)
        else:
            self.cash = float(self.initial_cash)
            self.shares = np.zeros(self.n, dtype=np.float64)

        self.prev_value = self.cash + float(np.sum(self.shares * self.prices[self.t]))
        self.returns_history.clear()
        return self._obs(), {}

    def step(self, action):
        prices = self.prices[self.t]
        portfolio_value = self.cash + float(np.sum(self.shares * prices))

        #: turn the n raw scores plus an implicit 0-score for cash into
        #: portfolio weights (summing to 1 across tickers + cash) via softmax,
        #: then translate each ticker's weight into a target position. The
        #: whole portfolio is redistributed every step — no fixed per-ticker
        #: budgets — and only the (possibly fractional) difference between the
        #: current and target position is traded.
        scores = np.append(np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0), 0.0)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        target_values = weights[:self.n] * portfolio_value
        target_shares = np.divide(target_values, prices, out=np.zeros(self.n), where=prices > 0)
        diff_shares = target_shares - self.shares

        #: sell first to free up cash, then spend it on buys (capped by what's available)
        for i in np.where(diff_shares < 0)[0]:
            sell_shares = min(-diff_shares[i], self.shares[i])
            self.cash += sell_shares * prices[i]
            self.shares[i] -= sell_shares
        for i in np.where(diff_shares > 0)[0]:
            cost = min(diff_shares[i] * prices[i], self.cash)
            self.shares[i] += cost / prices[i]
            self.cash -= cost

        portfolio_value = self.cash + float(np.sum(self.shares * prices))

        #: reward defined as a rolling Sharpe ratio of step returns, rather than
        #: raw returns, so the agent is pushed towards consistent risk-adjusted gains.
        step_return = (portfolio_value - self.prev_value) / self.prev_value if self.prev_value > 0 else 0.0
        self.returns_history.append(step_return)
        self.prev_value = portfolio_value

        returns_arr = np.array(self.returns_history, dtype=np.float64)
        std = returns_arr.std()
        if len(returns_arr) >= 2 and std > 0:
            reward = float(returns_arr.mean() / std)
        else:
            reward = 0.0

        #: t is the upperbound of the time window.
        self.t += 1
        terminated = bool(self.t >= len(self.prices))

        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, False, {"portfolio_value": portfolio_value}

    def _obs(self):
        window_features = self.feature_data[self.t - self.window : self.t]
        flat_features = window_features.transpose(1, 0, 2).flatten()  # (n * window * 4,)

        corr = self.corr_data[self.t]  # (n_pairs,)
        prices = self.prices[self.t]
        portfolio_value = self.cash + float(np.sum(self.shares * prices))
        #: each ticker's share of total portfolio value (can be partial)
        position_ratios = (self.shares * prices / portfolio_value).astype(np.float32)
        cash_ratio = np.float32(self.cash / portfolio_value)

        return np.concatenate([flat_features, corr, position_ratios, [cash_ratio]]).astype(np.float32)

    def render(self):
        prices = self.prices[self.t - 1]
        value = self.cash + float(np.sum(self.shares * prices))
        positions = " | ".join(
            f"{self.tickers[i]}={self.shares[i] * prices[i] / value:.0%}"
            for i in range(self.n)
        )
        positions += f" | CASH={self.cash / value:.0%}"
        corr = self.corr_data[self.t - 1]
        pairs = list(combinations(range(self.n), 2))
        corr_str = " | ".join(
            f"ρ({self.tickers[i]},{self.tickers[j]})={corr[k]:.2f}"
            for k, (i, j) in enumerate(pairs)
        )
        print(f"t={self.t:4d} | value={value:10.2f} | {positions} | {corr_str}")
