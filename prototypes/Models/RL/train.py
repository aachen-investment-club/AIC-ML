import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import TradingEnv



TICKERS     = ["AAPL", "MSFT", "GLD"]
TRAIN_START = "2015-01-01" #: trained on bullish market. 
TRAIN_END   = "2022-01-01"

EVAL_START  = "2022-01-01" 
EVAL_END    = "2023-01-01"


def load(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]]
    close = close.dropna()
    #: only use close data
    print(close.columns)
    print(close.index)
    return {ticker: close[ticker] for ticker in tickers}



def train():
    price_data = load(TICKERS, TRAIN_START, TRAIN_END)
    env = TradingEnv(price_data)
    check_env(env, warn=True)

    model = PPO("MlpPolicy", env,
                 verbose=1,
                   device="cuda",
                     ent_coef=0.01, n_steps=4096, batch_size=256)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_trading")
    print("Model saved to ppo_trading.zip")
    return model


def evaluate(model):
    price_data = load(TICKERS, EVAL_START, EVAL_END)
    env = TradingEnv(price_data, randomize_start=False)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        env.render()

    final = info["portfolio_value"]
    baseline = sum(
        price_data[t].iloc[-1] / price_data[t].iloc[0] * (10_000 / len(TICKERS))
        for t in TICKERS
    )
    print(f"\nFinal portfolio : ${final:,.2f}")
    print(f"Equal-weight buy-and-hold : ${baseline:,.2f}")


if __name__ == "__main__":
    load(TICKERS, TRAIN_START, TRAIN_END)
    #model = train()
    #evaluate(model)
