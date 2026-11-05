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

from strategies.models.svm import SVMStrategy
from strategies.interfaces import TradeLog

params = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "auto",
    "confidence_threshold": 0.53
}

def train_test():
    SVMStrategy.hyperparams = params
    
    # 1. Fetch training data from SQLite database
    print("[SVM Test] Fetching training data...")
    SVMStrategy.get_training_data()

    # 2. Extract and scale features
    print("[SVM Test] Extracting and scaling features...")
    SVMStrategy.extract_features()

    # 3. Train the SVM classifier
    print("[SVM Test] Training model...")
    SVMStrategy.train()

    # 4. Run the backtest on testing set
    print("[SVM Test] Running backtest...")
    SVMStrategy.test()

def execute(): 
    print("[SVM Test] Loading trained model...")
    SVMStrategy.load_model()
    
    # Try fetching tradelog from S3, fallback to local empty TradeLog if S3 fails (e.g. offline/no credentials)
    print("[SVM Test] Fetching trade log...")
    try:
        SVMStrategy.get_tradelog()
    except Exception as e:
        print(f"[SVM Test] S3 get_tradelog failed ({e}). Initializing empty local trade log.")
        SVMStrategy.tradelog = TradeLog(transactions=[], version="1.0", name=SVMStrategy.strategy_name)
        
    today = datetime.now() - timedelta(days=1)
    
    # Simulate signals for the last 9 days
    for i in range(8, -1, -1):
        sim_date = today - timedelta(days=i)
        
        if sim_date.weekday() >= 5:
            print(f"[{sim_date.strftime('%Y-%m-%d')}] Weekend - Market Closed.")
            continue

        try:
            signal = SVMStrategy.trade(ticker="AAPL", target_date=sim_date)
            action = "LONG (1)" if signal == 1 else "FLAT (0)"
            
            if i == 0:
                print(f"[TODAY - {sim_date.strftime('%Y-%m-%d')}] Live Signal for AAPL: {action}")
            else:
                print(f"[{sim_date.strftime('%Y-%m-%d')}] Simulated Signal for AAPL: {action}")
                
        except Exception as e:
             print(f"[{sim_date.strftime('%Y-%m-%d')}] Failed to generate signal: {e}")

    print("\n[SVM Test] Current Trade Log:")
    print(SVMStrategy.tradelog)
    
    print("[SVM Test] Uploading trade log...")
    try:
        SVMStrategy.upload_tradelog()
        print("[SVM Test] Trade log successfully uploaded to S3.")
    except Exception as e:
        print(f"[SVM Test] S3 upload_tradelog failed ({e}). Ignoring upload.")

if __name__ == "__main__": 
    print("=== STARTING SVM STRATEGY TRAINING AND BACKTEST ===")
    train_test()
    
    print("\n=== STARTING SVM STRATEGY LIVE TRADING SIMULATION ===")
    execute()
    print("=== SVM STRATEGY TESTING COMPLETE ===")
