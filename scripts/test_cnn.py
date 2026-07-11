"""
Run with:

python scripts/test_cnn.py       

"""
import sys
from datetime import datetime, timedelta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)

portfolio_repo_path = os.path.join(base_dir, "portfolio-management")
if portfolio_repo_path not in sys.path:
    sys.path.append(portfolio_repo_path)


from dotenv import load_dotenv

load_dotenv()

from portfolio.utils.aws_config import engine
from strategies.models.cnn import CNNStrategy


params = {
    "conv_filters": 64,
    "kernel_size": 3,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "patience": 5,
}


def train_test():
    """Train and test the CNN model."""
    CNNStrategy.hyperparams = params
    CNNStrategy.get_training_data()
    CNNStrategy.extract_features()
    CNNStrategy.train()
    CNNStrategy.test()


def execute():
    """Execute live trading simulation."""
    CNNStrategy.load_model()
    print("Fetching tradelog...")
    CNNStrategy.get_tradelog()
    today = datetime.now() - timedelta(days=1)

    for i in range(8, -1, -1):
        sim_date = today - timedelta(days=i)

        if sim_date.weekday() >= 5:
            print(f"[{sim_date.strftime('%Y-%m-%d')}] Weekend - Market Closed.")
            continue

        try:
            signal = CNNStrategy.trade(ticker="AAPL", target_date=sim_date)

            action = "LONG (1)" if signal == 1 else "FLAT (0)"

            if i == 0:
                print(f"[TODAY - {sim_date.strftime('%Y-%m-%d')}] Live Signal for AAPL: {action}")
            else:
                print(f"[{sim_date.strftime('%Y-%m-%d')}] Simulated Signal for AAPL: {action}")

        except Exception as e:
            print(f"[{sim_date.strftime('%Y-%m-%d')}] Failed to generate signal: {e}")

    print(CNNStrategy.tradelog)
    CNNStrategy.upload_tradelog()


if __name__ == "__main__":
    train_test()
    execute()
