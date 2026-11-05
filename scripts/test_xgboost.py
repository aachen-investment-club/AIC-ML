import sys
from datetime import datetime, timedelta
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)

portfolio_repo_path = os.path.join(base_dir, "portfolio-management")
if portfolio_repo_path not in sys.path:
    sys.path.append(portfolio_repo_path)

from dotenv import load_dotenv
load_dotenv()

from strategies.models.xgboost import XGBoostStrategy
from strategies.interfaces import TradeLog

params = {
    "n_estimators": 50,
    "max_depth": 7,
    "learning_rate": 0.01,
    "confidence_threshold": 0.50
}

def train_test():
    XGBoostStrategy.hyperparams = params
    
    # 1. Fetch training data from SQLite database
    print("[XGBoost Test] Fetching training data...")
    XGBoostStrategy.get_training_data()

    # 2. Extract and scale features
    print("[XGBoost Test] Extracting and scaling features...")
    XGBoostStrategy.extract_features()

    # 3. Train the XGBoost classifier
    print("[XGBoost Test] Training model...")
    XGBoostStrategy.train()

    # 4. Run the backtest on testing set
    print("[XGBoost Test] Running backtest...")
    XGBoostStrategy.test()

def execute(): 
    print("[XGBoost Test] Loading trained model...")
    XGBoostStrategy.load_model()
    
    # Try fetching tradelog from S3, fallback to local empty TradeLog if S3 fails
    print("[XGBoost Test] Fetching trade log...")
    try:
        XGBoostStrategy.get_tradelog()
    except Exception as e:
        print(f"[XGBoost Test] S3 get_tradelog failed ({e}). Initializing empty local trade log.")
        XGBoostStrategy.tradelog = TradeLog(transactions=[], version="1.0", name=XGBoostStrategy.strategy_name)
        
    today = datetime.now() - timedelta(days=1)
    
    # Simulate signals for the last 9 days
    for i in range(8, -1, -1):
        sim_date = today - timedelta(days=i)
        
        if sim_date.weekday() >= 5:
            print(f"[{sim_date.strftime('%Y-%m-%d')}] Weekend - Market Closed.")
            continue

        try:
            signal = XGBoostStrategy.trade(ticker="AAPL", target_date=sim_date)
            action = "LONG (1)" if signal == 1 else "FLAT (0)"
            
            if i == 0:
                print(f"[TODAY - {sim_date.strftime('%Y-%m-%d')}] Live Signal for AAPL: {action}")
            else:
                print(f"[{sim_date.strftime('%Y-%m-%d')}] Simulated Signal for AAPL: {action}")
                
        except Exception as e:
             print(f"[{sim_date.strftime('%Y-%m-%d')}] Failed to generate signal: {e}")

    print("\n[XGBoost Test] Current Trade Log:")
    print(XGBoostStrategy.tradelog)
    
    print("[XGBoost Test] Uploading trade log...")
    try:
        XGBoostStrategy.upload_tradelog()
        print("[XGBoost Test] Trade log successfully uploaded to S3.")
    except Exception as e:
        print(f"[XGBoost Test] S3 upload_tradelog failed ({e}). Ignoring upload.")

if __name__ == "__main__": 
    print("=== STARTING XGBOOST STRATEGY TRAINING AND BACKTEST ===")
    train_test()
    
    print("\n=== STARTING XGBOOST STRATEGY LIVE TRADING SIMULATION ===")
    execute()
    print("=== XGBOOST STRATEGY TESTING COMPLETE ===")
