"""
run with 

python scripts/test_lstm.py       

"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)

# 3. Add the nested portfolio_management folder to the path (so 'portfolio' imports work)
portfolio_repo_path = os.path.join(base_dir, "portfolio_management")
if portfolio_repo_path not in sys.path:
    sys.path.append(portfolio_repo_path)


from dotenv import load_dotenv

load_dotenv()

from portfolio.utils.aws_config import engine
from strategies.models.lstm import LSTMStrategy





if __name__ == "__main__": 

    LSTMStrategy.get_training_data()
    #print(LSTMStrategy.df.tail())
    print(LSTMStrategy.df.shape)

    LSTMStrategy.extract_features()

    #print(LSTMStrategy.X_train)
    #print(LSTMStrategy.X_test)
    #print(LSTMStrategy.y_train)
    #print(LSTMStrategy.y_test)

    LSTMStrategy.train()

    LSTMStrategy.trade()

