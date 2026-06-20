import pandas as pd
import numpy as np
from analyser import analyse 

# 1. Kursdaten für 5 Tage
ticker_daten = pd.DataFrame({
    'date':  ['2026-06-01', '2026-06-02', '2026-06-03', '2026-06-04', '2026-06-05'],
    'close': [100.0,        105.0,        110.0,        115.0,        120.0]
})

# 2. Test-Trades (Uhrzeit auf 00:00:00 gesetzt, damit der Abgleich klappt)
trade_daten = pd.DataFrame({
    'date':   ['2026-06-01', '2026-06-05'],
    'time':   ['00:00:00',   '00:00:00'], 
    'type':   ['PURCHASE',   'SALE'],
    'shares': [100.0,        100.0]
})

# ========================================================
# HIER SIND DIE PRINTS JETZT AN DER RICHTIGEN STELLE:
# ========================================================
print("Spalten in ticker_data:", ticker_daten.columns.tolist())
print("Spalten in trade_logs:",  trade_daten.columns.tolist())
print("-" * 50)

# Aufruf deiner Funktion
ergebnis_sharpe = analyse(ticker_daten, trade_daten, risk_free_rate=0.0)

print("=" * 40)
print(f"Der berechnete Sharpe Ratio ist: {ergebnis_sharpe}")
print("=" * 40)
