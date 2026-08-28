import sys
import os
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

portfolio_repo_path = os.path.join(base_dir, "portfolio-management")
if portfolio_repo_path not in sys.path:
    sys.path.append(portfolio_repo_path)

from dotenv import load_dotenv
load_dotenv()

# Force local day-level database
os.environ["DB_GRANULARITY"] = "day"

from portfolio.models.market import Market
from portfolio.schemas.market import Base, MarketDB, ForexDayDB, TickerMeta
from portfolio.utils.aws_config import engine

def main():
    print("Creating tables in SQLite database...")
    Base.metadata.create_all(engine)
    
    print("Populating ticker metadata...")
    if Market.check_meta_empty():
        df = pd.read_csv("portfolio-management/data/ticker_metadata.csv")
        df = df.rename(columns={"symbol": "ticker"})
        df = df.drop_duplicates(subset=["ticker"])
        df.to_sql(TickerMeta.__tablename__, con=engine, if_exists="append", index=False)
        print(f"Loaded {len(df)} metadata rows.")
    else:
        print("Ticker metadata already exists.")

    print("Populating market data...")
    if Market.check_empty():
        # Load market data in chunks or fully using pandas to_sql
        chunksize = 100000
        total = 0
        for chunk in pd.read_csv("portfolio-management/data/market_export.csv", chunksize=chunksize):
            chunk = chunk.rename(columns={
                "Ticker": "ticker",
                "Date": "date",
                "Price Close": "price_close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume"
            })
            chunk["date"] = pd.to_datetime(chunk["date"]).dt.floor("D")
            chunk = chunk.drop_duplicates(subset=["ticker", "date"], keep="last")
            chunk.to_sql(MarketDB.__tablename__, con=engine, if_exists="append", index=False)
            total += len(chunk)
            print(f"Inserted {total} market rows...")
        print("Market data populated.")
    else:
        print("Market data already exists.")

    print("Populating forex data...")
    if Market.check_forex_empty():
        # Load forex data
        df = pd.read_csv("portfolio-management/data/forex_export.csv")
        df = df.rename(columns={
            "Ticker": "ticker",
            "Date": "date",
            "Price Close": "price_close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
        df.to_sql(ForexDayDB.__tablename__, con=engine, if_exists="append", index=False)
        print(f"Loaded {len(df)} forex rows.")
    else:
        print("Forex data already exists.")

    print("Database initialization complete!")

if __name__ == "__main__":
    main()
