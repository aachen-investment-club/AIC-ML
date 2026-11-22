"""
run with

python scripts/run_weekly_evaluation.py
python scripts/run_weekly_evaluation.py --dry-run     # no upload to S3

"""
import argparse
import sys
from datetime import datetime, timedelta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)  # training_outputs/ and DB_PATH are relative to the repo root

if base_dir not in sys.path:
    sys.path.append(base_dir)

portfolio_repo_path = os.path.join(base_dir, "portfolio-management")
if portfolio_repo_path not in sys.path:
    sys.path.append(portfolio_repo_path)


from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

from strategies.interfaces import TradeLog
from strategies.models.lstm import LSTMStrategy
from strategies.models.svm import SVMStrategy
from strategies.models.xgboost import XGBoostStrategy


ACTIVE_STRATEGIES = [XGBoostStrategy, LSTMStrategy, SVMStrategy]
TICKER = "AAPL"
LOOKBACK_DAYS = 7


def load_tradelog(strategy):
    try:
        strategy.get_tradelog()
    except ClientError as e:
        # only start a new log if none exists; the upload would overwrite the history otherwise
        if e.response["Error"]["Code"] != "NoSuchKey":
            raise
        strategy.tradelog = TradeLog(transactions=[], version="1.0", name=strategy.strategy_name)


def evaluate(strategy, dry_run):
    load_tradelog(strategy)
    logged_dates = {t["date"][:10] for t in strategy.tradelog.transactions}
    new_trades = 0

    for i in range(LOOKBACK_DAYS, 0, -1):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        if date.weekday() >= 5 or date_str in logged_dates:
            continue

        # holiday or outdated DB: don't trade on an old close
        if strategy.get_data_for_trade(TICKER, target_date=date).index[-1].date() != date.date():
            print(f"[{strategy.strategy_name}] {date_str}: no market data, skipping.")
            continue

        signal = strategy.trade(ticker=TICKER, target_date=date)
        print(f"[{strategy.strategy_name}] {date_str}: {'LONG (1)' if signal == 1 else 'FLAT (0)'}")
        new_trades += 1

    if new_trades and not dry_run:
        strategy.upload_tradelog()
    print(f"[{strategy.strategy_name}] Done: {new_trades} new trades.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="skip the upload to S3")
    args = parser.parse_args()

    failed = []
    for strategy in ACTIVE_STRATEGIES:
        try:
            evaluate(strategy, args.dry_run)
        except Exception as e:
            print(f"[{strategy.strategy_name}] FAILED: {e}")
            failed.append(strategy.strategy_name)

    if failed:
        sys.exit(f"Failed strategies: {', '.join(failed)}")
