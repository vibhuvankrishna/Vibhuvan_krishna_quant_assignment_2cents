from ib_insync import *
import pandas as pd

# --- Connect ---
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# --- Define BTC/USD Contract ---
contract = Crypto('BTC', 'PAXOS', 'USD')  # or 'CRYPTO' if PAXOS fails

# --- Request Historical Data ---
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',          # 1 day of data
    barSizeSetting='1 min',
    whatToShow='AGGTRADES',     # ✅ required for crypto
    useRTH=False,
    formatDate=1
)

# --- Convert & Display ---
if bars:
    df = util.df(bars)
    print(df.head())
    df.to_csv('BTCUSD_1min_IBKR.csv', index=False)
    print("✅ Saved to BTCUSD_1min_IBKR.csv")
else:
    print("⚠️ No data returned. You may not have crypto data permissions in trial mode.")

ib.disconnect()
