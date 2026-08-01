"""
Rank the FULL Nifty 100 pool (union of NIFTY_50 + NIFTY_NEXT_50 from
`ema_regime_crossover_swing_cnc_nifty100_backtest.py`, 99 unique names) by
trailing realized daily-return volatility (annualized stdev of daily
close-to-close log returns, causal/historical only -- no forward-looking
data), to test whether the ORIGINAL 59-name pool used in
`ema_regime_crossover_high_beta_rank.py` (test #8's Nifty 50 + a fixed
top-10 Nifty Next 50 subset) was excluding genuinely higher-beta names that
would make a better top-15 than the currently-deployed HIGH_BETA_15.

Method: identical to `ema_regime_crossover_high_beta_rank.py` -- daily
close-to-close log returns over the full backtest history (2021-07-01 to
2026-06-30), annualized stdev = daily_stdev * sqrt(252). Single full-sample
ranking (not rolling), same rationale: used only to pick a fixed universe
subset for a backtest over the SAME historical window, not for live
day-by-day decisions. Uses only price history, not trade outcomes -- not
overfit to the strategy's own signals.

ONLY the source pool changes vs. the original ranking script -- everything
else (window, method, output format) is unchanged.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE = "2026-06-30"
IST = pytz.timezone("Asia/Kolkata")

# Full Nifty 100 pool -- verbatim from
# ema_regime_crossover_swing_cnc_nifty100_backtest.py (sourced 2026-07-17,
# tickertape.in Nifty 50 / Nifty Next 50 pages).
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

_seen = set()
UNIVERSE = []
for _sym in NIFTY_50 + NIFTY_NEXT_50:
    if _sym not in _seen:
        _seen.add(_sym)
        UNIVERSE.append(_sym)

from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)


def _fetch(symbol, interval):
    try:
        resp = client.history(symbol=symbol, exchange="NSE", interval=interval,
                               start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        return None, f"error: {e}"
    if isinstance(resp, dict):
        if resp.get("status") != "success":
            return None, f"api error: {resp.get('message', resp)}"
        df = pd.DataFrame(resp.get("data", []))
        if df.empty:
            return None, "no data"
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        df = resp
        if df is None or df.empty:
            return None, "no data"
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert(IST)
    return df.sort_index(), None


rows = []
skipped = {}
print(f"Universe: {len(UNIVERSE)} names (full Nifty 100 pool)")
for symbol in UNIVERSE:
    dfd, errd = _fetch(symbol, "D")
    if errd:
        skipped[symbol] = errd
        print(f"  {symbol}: SKIPPED ({errd})")
        continue
    ret = np.log(dfd["close"] / dfd["close"].shift(1)).dropna()
    if len(ret) < 60:
        skipped[symbol] = f"too few daily bars ({len(ret)})"
        continue
    ann_vol = ret.std() * np.sqrt(252) * 100
    atr_pct_proxy = (dfd["high"] - dfd["low"]).abs().div(dfd["close"]).mean() * 100
    rows.append({"symbol": symbol, "n_days": len(ret), "ann_vol_pct": round(ann_vol, 2),
                 "avg_daily_range_pct": round(atr_pct_proxy, 3)})

rank_df = pd.DataFrame(rows).sort_values("ann_vol_pct", ascending=False).reset_index(drop=True)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_high_beta_rank_nifty100")
os.makedirs(OUT_DIR, exist_ok=True)
rank_df.to_csv(os.path.join(OUT_DIR, "high_beta_ranking_nifty100.csv"), index=False)

print(f"\nSkipped: {skipped}")
print(f"\nRanked {len(rank_df)} names by annualized realized-return volatility (2021-07-01 -> 2026-06-30):")
print(rank_df.to_string(index=False))

top15 = rank_df.head(15)["symbol"].tolist()
top12 = rank_df.head(12)["symbol"].tolist()
top10 = rank_df.head(10)["symbol"].tolist()
print(f"\nTop 10 : {top10}")
print(f"Top 12 : {top12}")
print(f"Top 15 : {top15}")

DEPLOYED_HIGH_BETA_15 = [
    "ADANIGREEN", "ADANIENSOL", "ADANIENT", "ADANIPOWER", "ETERNAL", "LODHA",
    "TMCV", "VEDL", "ADANIPORTS", "CGPOWER", "MAXHEALTH", "CANBK",
    "SHRIRAMFIN", "TRENT", "JIOFIN",
]
same = [s for s in top15 if s in DEPLOYED_HIGH_BETA_15]
new = [s for s in top15 if s not in DEPLOYED_HIGH_BETA_15]
dropped = [s for s in DEPLOYED_HIGH_BETA_15 if s not in top15]
print(f"\nSame as deployed (survived wider pool): {same}")
print(f"New (not previously considered / not in top15 under old pool): {new}")
print(f"Dropped from deployed list (now outside new top15): {dropped}")
