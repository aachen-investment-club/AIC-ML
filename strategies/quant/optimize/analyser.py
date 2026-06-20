import pandas as pd #Tabellenlogik
import numpy as np #sqrt log usw usw

# pd.DataFrame is tabelle type
def analyse(ticker_data: pd.DataFrame,trade_logs: pd.DataFrame, risk_free_rate: float = 0.0)-> float:

    # datum und uhrzeit zusammen in spalte
    ticker_df = ticker_data.copy()
    trade_df = trade_logs.copy()

    ticker_df['datetime'] = pd.to_datetime(ticker_df['date'])#macht eintrag einheitlich zu datetime typ
    ticker_df = ticker_df.sort_values('datetime').reset_index(drop=True)

    trade_df['datetime'] = pd.to_datetime(trade_df['date'].astype(str) + ' ' + trade_df['time'].astype(str))

    # portfolio-simulation
    #rechnet tag für tag von vergangenheit bankkonto aus (pro tag cash shares und total values)
    initial_cash = 100000.0
    current_cash = initial_cash #Ct
    current_shares = 0.0 #AnzahlAktien
    
    returns_history = []
    full_return = 0.0
    current_return = 0.0
    anzahl_perioden = 0
    average_return = 0.0


    price_map = dict(zip(ticker_df['datetime'], ticker_df['close'])) # key:datetime value:close


    last_total_value = initial_cash


    for current_time in ticker_df['datetime']: # für alle datums
        current_price = price_map.get(current_time, 0.0)# nimmt preis vom aktuellen datum
        matching_trades = trade_df[trade_df['datetime'].dt.date == current_time.date()] #trades für aktuelles datum

        for _, trade in matching_trades.iterrows(): # vom aktuellen datum alle trades (index=0, trade = {"type": "PURCHASE", "shares": 150.0} )
            trade_value = float(trade['shares']) * current_price

            if trade['type'] == "PURCHASE":
                current_cash   -= trade_value
                current_shares += float(trade['shares'])
                
            elif trade['type'] == "SALE":
                current_cash += trade_value
                current_shares -= float(trade['shares'])

        total_value = current_cash +(current_shares * current_price)

       
        if anzahl_perioden > 0:
            current_return = (total_value - last_total_value)/last_total_value
            returns_history.append(current_return)
            full_return = full_return + current_return


        last_total_value = total_value
        anzahl_perioden += 1

    if anzahl_perioden <= 1:
        return 0.0
    
    #sigma
    average_return = full_return / (anzahl_perioden-1)
    d = 0.0
    f = 0
    for p in returns_history:
        d += (p - average_return)**2
        f += 1

    sigma = np.sqrt(d/(f-1)) if f > 1 else 0.0

    if sigma > 0:
        sharpe_ratio = (average_return - risk_free_rate) / sigma
    else:
        sharpe_ratio = 0.0
   

    return float(sharpe_ratio)

    









    #rendite berechnung

    #


