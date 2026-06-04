import pandas as pd


def input_adapter(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    mapping = {
        "ticker": "ticker",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "price_close": "close",
        "volume": "volume",
    }
    data = data.rename(mapping)
    
    return data


def output_adapter(
        signal_df: pd.DataFrame, 
        meta_df: pd.DataFrame, 
        portfolio_name: str = "AIC to the moon",
        shares_to_trade: float = 150.0
) -> dict:
    transactions = []

    # Identify where the trading position alters state
    signal_df["prev_signal"] = signal_df["signal"].shift(1).fillna(0)
    action_rows = signal_df[signal_df["signal"] != signal_df["prev_signal"]]

    for _,row in action_rows.iterrows():
        ticker = row["ticker"]
        current_signal = int(row["signal"])
        prev_signal = int(row["prev_signal"])

        meta_row = meta_df[meta_df["ticker"] == ticker]
        if meta_row.empty:
            long_name = f"{ticker} Corporation"
            currency = "USD"
        else:
            long_name = str(meta_row["longname"].values[0])
            currency = str(meta_row["currency"].values[0])

        # Determine transaction type based on position state transitions
        tx_type = None
        if prev_signal == 0 and current_signal == 1:
            tx_type = "PURCHASE"
        elif prev_signal == 0 and current_signal == -1:
            tx_type = "SALE"
        
        if not tx_type:
            continue

        # Format the timestamp
        dt_obj = pd.to_datetime(row["date"])
        date_str = dt_obj.strftime("%Y-%m-%d")
        time_str = dt_obj.strftime("%H:%M")

        tx_node = {
            "type": tx_type,
            "type": tx_type,
            "account": currency,
            "portfolio": portfolio_name,
            "date": date_str,
            "time": time_str,
            "currency": currency,
            "shares": float(shares_to_trade),
            "security": {
                "name": long_name,
                "ticker": ticker,
                "currency": currency
            }
        }
        transactions.append(tx_node)

    return {
        "version": 1,
        "transactions": transactions
    }


    pass