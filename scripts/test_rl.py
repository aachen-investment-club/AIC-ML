
"""
run with 

python scripts/test_rl.py       

"""
import sys
from datetime import datetime, timedelta

import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)

portfolio_repo_path = os.path.join(base_dir, "portfolio_management")

if portfolio_repo_path not in sys.path:
    sys.path.append(portfolio_repo_path)


from dotenv import load_dotenv

load_dotenv()

from portfolio.utils.aws_config import engine
from strategies.models.RL_PPO.reinforcement_learning_strategy_1 import RLStrategy

def train_test(): 
    RLStrategy.get_training_data()
    RLStrategy.train()
    RLStrategy.test()




if __name__=="__main__": 
    train_test()

