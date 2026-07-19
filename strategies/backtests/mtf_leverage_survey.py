"""
Real Dhan MTF leverage survey — Nifty 100 (Nifty 50 + Nifty Next 50)

Karan noticed the Sandbox equity_mtf_leverage default (2x, a conservative
placeholder set 2026-07-19 when MTF support was first added) doesn't match
reality -- Dhan's real MTF leverage for SBIN came back at 4.55x. This script
queries Dhan's REAL margin calculator (via OpenAlgo's own /api/v1/margin,
which now supports product=MTF end-to-end) for every Nifty 100 name, to set
an evidence-based default instead of a guess.

Method: fetch each symbol's live LTP (multiquotes), then call /api/v1/margin
with a 1-share BUY, product=MTF, pricetype=MARKET at that LTP. Dhan's real
margincalculator API returns total_margin_required for that exact position.
implied_leverage = trade_value / total_margin_required (trade_value = LTP x 1
share, so this is just LTP / total_margin_required).

Uses the same Nifty 50 / Nifty Next 50 constituent lists already sourced
2026-07-17 for the EMA Regime Crossover universe (universe_ranking.py) --
not re-derived here.
"""

import os
import time

import pandas as pd
import requests

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

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

UNIVERSE = NIFTY_50 + NIFTY_NEXT_50

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_multiquotes(symbols):
    """symbols: list of (symbol, exchange) tuples. Returns {symbol: ltp}."""
    r = requests.post(
        f"{HOST}/api/v1/multiquotes",
        json={"apikey": API_KEY, "symbols": [{"symbol": s, "exchange": e} for s, e in symbols]},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    out = {}
    for res in data["results"]:
        try:
            out[res["symbol"]] = float(res["data"]["ltp"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def get_mtf_margin(symbol, ltp):
    """Real Dhan margin calculator, product=MTF, 1-share BUY at current LTP."""
    r = requests.post(
        f"{HOST}/api/v1/margin",
        json={
            "apikey": API_KEY,
            "positions": [{
                "exchange": "NSE", "symbol": symbol, "action": "BUY",
                "quantity": "1", "product": "MTF", "pricetype": "MARKET", "price": str(ltp),
            }],
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


print(f"Fetching live LTPs for {len(UNIVERSE)} names...")
ltp_map = get_multiquotes([(s, "NSE") for s in UNIVERSE])
print(f"Got LTPs for {len(ltp_map)}/{len(UNIVERSE)} names")

rows = []
for symbol in UNIVERSE:
    ltp = ltp_map.get(symbol)
    if ltp is None:
        rows.append({"symbol": symbol, "ltp": None, "margin_required": None,
                      "implied_leverage": None, "error": "no LTP"})
        continue
    try:
        resp = get_mtf_margin(symbol, ltp)
        if resp.get("status") != "success":
            rows.append({"symbol": symbol, "ltp": ltp, "margin_required": None,
                          "implied_leverage": None, "error": resp.get("message", "margin call failed")})
            print(f"  {symbol}: FAILED ({resp.get('message', resp)})")
        else:
            margin = float(resp["data"]["total_margin_required"])
            leverage = (ltp / margin) if margin > 0 else None
            rows.append({"symbol": symbol, "ltp": ltp, "margin_required": margin,
                          "implied_leverage": leverage, "error": None})
            print(f"  {symbol}: LTP={ltp:.2f} margin={margin:.2f} leverage={leverage:.2f}x" if leverage else f"  {symbol}: margin=0")
    except Exception as e:
        rows.append({"symbol": symbol, "ltp": ltp, "margin_required": None,
                      "implied_leverage": None, "error": str(e)})
        print(f"  {symbol}: ERROR ({e})")
    time.sleep(0.35)  # gentle on Dhan's real margin-calculator endpoint

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "mtf_leverage_survey.csv"), index=False)

valid = df.dropna(subset=["implied_leverage"])
print(f"\n=== MTF Leverage Survey: {len(valid)}/{len(df)} names succeeded ===")
if not valid.empty:
    print(f"Mean:   {valid['implied_leverage'].mean():.2f}x")
    print(f"Median: {valid['implied_leverage'].median():.2f}x")
    print(f"Min:    {valid['implied_leverage'].min():.2f}x ({valid.loc[valid['implied_leverage'].idxmin(), 'symbol']})")
    print(f"Max:    {valid['implied_leverage'].max():.2f}x ({valid.loc[valid['implied_leverage'].idxmax(), 'symbol']})")
    print(f"Std:    {valid['implied_leverage'].std():.2f}")
    print("\nFull distribution (sorted):")
    print(valid[["symbol", "ltp", "margin_required", "implied_leverage"]].sort_values("implied_leverage").to_string(index=False))

failed = df[df["error"].notna()]
if not failed.empty:
    print(f"\n{len(failed)} names failed:")
    print(failed[["symbol", "error"]].to_string(index=False))
