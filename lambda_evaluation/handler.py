"""
weekly evaluation of the latest models in S3, run with

python -m lambda_evaluation.handler            # dry run, nothing is uploaded
python -m lambda_evaluation.handler --upload   # uploads the tradelogs to S3
"""
import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta

import boto3
import pandas as pd
import yfinance as yf
from botocore.exceptions import ClientError


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.getenv("WORK_DIR", "/tmp/weekly_evaluation")   # /tmp is the only writable folder in lambda

TICKER = "AAPL"
LOOKBACK_DAYS = 7
HISTORY_DAYS = 220   # LSTM builds its features from the last 200 days


def active_strategies():
    from strategies.models.lstm import LSTMStrategy
    return [LSTMStrategy]


def add_import_paths():
    candidates = [
        REPO_DIR,
        os.path.join(REPO_DIR, "portfolio-management"),
        os.path.join(os.path.dirname(REPO_DIR), "portfolio-management"),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def download_latest_model(strategy_name: str, s3_client=None) -> list:
    s3_client = s3_client or boto3.client("s3")
    bucket = os.getenv("MODEL_ARTIFACT_BUCKET") or "portfolio-management-developer"
    prefix = f"models/{strategy_name}/latest/"
    target_dir = os.path.join(WORK_DIR, "training_outputs", strategy_name, "latest")

    keys = []
    for page in s3_client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [obj["Key"] for obj in page.get("Contents", []) if not obj["Key"].endswith("/")]
    if not keys:
        raise FileNotFoundError(f"no model found at s3://{bucket}/{prefix}")

    for key in keys:
        local_path = os.path.join(target_dir, *key[len(prefix):].split("/"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket, key, local_path)

    print(f"[{strategy_name}] Downloaded {len(keys)} files from s3://{bucket}/{prefix}")
    return keys


def load_daily_prices(tickers: list, days: int):
    yf.set_tz_cache_location(WORK_DIR)   # the default cache folder is read-only in lambda
    data = yf.download(
        tickers,
        start=datetime.now() - timedelta(days=days),
        end=datetime.now() + timedelta(days=1),
        interval="1d",
        group_by="ticker",
        progress=False,
    )

    frames = []
    for ticker in tickers:
        df = (data[ticker] if isinstance(data.columns, pd.MultiIndex) else data).dropna(subset=["Close"]).reset_index()
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": df["Date"], "price_close": df["Close"],
            "open": df["Open"], "high": df["High"], "low": df["Low"], "volume": df["Volume"],
        }))
    prices = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(prices)} daily prices for {tickers} from yfinance ({prices['date'].min().date()} -> {prices['date'].max().date()})")
    return prices


def build_local_db(prices) -> None:
    from portfolio.schemas.market import Base
    from portfolio.models.market import Market
    from portfolio.utils.aws_config import engine

    Base.metadata.create_all(engine)
    csv_path = os.path.join(WORK_DIR, "daily_prices.csv")
    prices.rename(columns={
        "ticker": "Ticker", "date": "Date", "price_close": "Price Close",
        "open": "Open", "high": "High", "low": "Low", "volume": "Volume",
    }).to_csv(csv_path, index=False)
    Market.load_from_csv(csv_path, "market", 1000)


def load_tradelog(strategy) -> None:
    from strategies.interfaces import TradeLog
    try:
        strategy.get_tradelog()
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            raise
        strategy.tradelog = TradeLog(transactions=[], version="1.0", name=strategy.strategy_name)


def evaluate(strategy, upload: bool, trading_days: set) -> list:
    load_tradelog(strategy)
    logged_dates = {t["date"][:10] for t in strategy.tradelog.transactions}
    signals = []

    for i in range(LOOKBACK_DAYS, 0, -1):
        day = datetime.now() - timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        if day.weekday() >= 5 or day_str in logged_dates:
            continue

        # holiday or missing data: don't trade on an old close
        if day.date() not in trading_days:
            print(f"[{strategy.strategy_name}] {day_str}: no market data, skipping.")
            continue

        signal = strategy.trade(ticker=TICKER, target_date=day)
        signals.append({"date": day_str, "signal": int(signal)})
        print(f"[{strategy.strategy_name}] {day_str}: {'LONG (1)' if signal == 1 else 'FLAT (0)'}")

    if signals and upload:
        strategy.upload_tradelog()
    return signals


def handler(event, context):
    """lambda entry point; pass {"upload": true} to upload the tradelogs"""
    upload = bool((event or {}).get("upload", False))

    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)
    # fresh prices and models on every run
    shutil.rmtree("training_outputs", ignore_errors=True)
    if os.path.exists("market.db"):
        os.remove("market.db")
    os.environ["DB_PATH"] = f"sqlite:///{WORK_DIR}/market.db"
    os.environ["DB_GRANULARITY"] = "day"
    add_import_paths()

    prices = load_daily_prices([TICKER], HISTORY_DAYS)
    build_local_db(prices)
    trading_days = set(prices["date"].dt.date)

    results = {}
    for strategy in active_strategies():
        try:
            download_latest_model(strategy.strategy_name)
            results[strategy.strategy_name] = evaluate(strategy, upload, trading_days)
        except Exception as e:
            print(f"[{strategy.strategy_name}] FAILED: {e}")
            results[strategy.strategy_name] = f"failed: {e}"

    print(f"Done (upload={upload}): {results}")
    return {"upload": upload, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="upload the tradelogs to S3")
    args = parser.parse_args()
    handler({"upload": args.upload}, None)
