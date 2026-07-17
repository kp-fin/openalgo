"""
universe_ranking.py — one-off utility to rank the current Nifty 50 and Nifty Next 50
constituents by trailing 20-day average daily traded value (ADV), for
equities-system/strategies/ema_regime_crossover.md's static-current-universe
backtest (top 10 of each pool).

Constituent lists sourced 2026-07-17 from tickertape.in (Nifty 50, Nifty Next 50
pages) since no NSE archive PDF or OpenAlgo tool provides a clean constituent list.
This is a point-in-time snapshot -- re-run if the universe needs refreshing later.
"""

import os
import time
from datetime import date, timedelta

import pandas as pd

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BAJAJFINSV", "AXISBANK", "BHARTIARTL",
    "BAJAJ-AUTO", "BAJFINANCE", "BEL", "ASIANPAINT", "DRREDDY", "EICHERMOT",
    "COALINDIA", "CIPLA", "HDFCBANK", "GRASIM", "HCLTECH", "HDFCLIFE", "HINDUNILVR",
    "ICICIBANK", "INFY", "HINDALCO", "INDIGO", "ITC", "JSWSTEEL", "M&M", "KOTAKBANK",
    "LT", "MARUTI", "MAXHEALTH", "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "SBILIFE",
    "SHRIRAMFIN", "SBIN", "RELIANCE", "TATASTEEL", "SUNPHARMA", "TCS", "TECHM",
    "TATACONSUM", "TMPV", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ETERNAL", "JIOFIN",
]

NIFTY_NEXT_50 = [
    "ABB", "ADANIPOWER", "AMBUJACEM", "ADANIENSOL", "ADANIGREEN", "BAJAJHLDNG",
    "BANKBARODA", "CGPOWER", "ZYDUSLIFE", "DLF", "BPCL", "CANBK", "DIVISLAB",
    "BRITANNIA", "CHOLAFIN", "CUMMINSIND", "DMART", "BOSCHLTD", "HAL", "GODREJCP",
    "GAIL", "HDFCAMC", "IOC", "HINDZINC", "INDHOTEL", "JINDALSTEL", "LTM",
    "UNITDSPR", "MUTHOOTFIN", "MOTHERSON", "RECLTD", "PIDILITIND", "PFC", "PNB",
    "TATAPOWER", "SOLARINDS", "SHREECEM", "SIEMENS", "UNIONBANK", "VEDL", "TVSMOTOR",
    "TORNTPHARM", "VBL", "MAZDOCK", "IRFC", "LODHA", "HYUNDAI", "ENRIN", "TATACAP",
    "TMCV",
]


END_DATE = date.today().isoformat()
START_DATE = (date.today() - timedelta(days=45)).isoformat()  # comfortably covers 20 trading days


def avg_daily_traded_value(symbol, lookback_days=20):
    try:
        resp = client.history(symbol=symbol, exchange="NSE", interval="D",
                               start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        return None, f"error: {e}"
    if isinstance(resp, dict):
        if resp.get("status") != "success":
            return None, f"api error: {resp}"
        df = pd.DataFrame(resp.get("data", []))
    else:
        df = resp
    if df is None or df.empty:
        return None, "no data"
    df.columns = [c.lower() for c in df.columns]
    df = df.tail(lookback_days)
    if df.empty or "volume" not in df.columns:
        return None, "insufficient data"
    traded_value = (df["close"] * df["volume"]).mean()
    return traded_value, None


def rank_pool(name, symbols):
    print(f"\nRanking {name} ({len(symbols)} names) by trailing 20-day ADV ...")
    rows = []
    for sym in symbols:
        adv, err = avg_daily_traded_value(sym)
        if adv is None:
            print(f"  {sym}: SKIPPED ({err})")
            continue
        rows.append({"symbol": sym, "adv": adv})
        time.sleep(0.05)  # gentle on the broker API
    ranked = pd.DataFrame(rows).sort_values("adv", ascending=False).reset_index(drop=True)
    return ranked


nifty50_ranked = rank_pool("Nifty 50", NIFTY_50)
next50_ranked = rank_pool("Nifty Next 50", NIFTY_NEXT_50)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
nifty50_ranked.to_csv(os.path.join(OUT_DIR, "nifty50_ranked.csv"), index=False)
next50_ranked.to_csv(os.path.join(OUT_DIR, "next50_ranked.csv"), index=False)

print("\n=== Top 10 Nifty 50 by 20-day ADV ===")
print(nifty50_ranked.head(10).to_string(index=False))
print("\n=== Top 10 Nifty Next 50 by 20-day ADV ===")
print(next50_ranked.head(10).to_string(index=False))
