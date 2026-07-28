"""
EMA Regime Crossover -- Baseline Backtest, Resized to Current Live Sizing

Identical to ema_regime_crossover_backtest.py (2026-07-17 original baseline) in
every respect EXCEPT position sizing: the original used a flat Rs 1 Cr / 12.5%
"OpenAlgo Sandbox convention" for backtest-to-backtest comparability, decoupled
from the live script's real-money sizing. This version instead uses the LIVE
script's current, real sizing formula (ema_regime_crossover_signal.py, corrected
2026-07-21 "capital-slice-then-leverage") -- allocated_capital (Rs 2,50,000,
paper mode, from strategies/scripts/state/capital_allocation.json) x 10%
pre-leverage slice x 5x ASSUMED_MIS_LEVERAGE for the full buying_power used to
size qty.

Exists solely as a fair comparison point for
ema_regime_crossover_trailing_target_backtest.py, which needed the same sizing
correction -- the much smaller per-trade notional at this capital scale changes
integer-share rounding enough that comparing the trailing-target result against
the original Rs-1-Cr-scale baseline CSV would conflate the sizing change with
the exit-rule change. Regime (EMA200/30m), entry (EMA9/20 crossover on 15m),
exit (1.36xATR stop / 3xATR target / reverse-cross / 15:00 hard exit), universe,
window, and max-6-concurrent cap are all unchanged from the original baseline.
"""

import os
import sys
import warnings
from datetime import time as dtime

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

REGIME_EMA = 200
ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14
STOP_ATR_MULT = 1.36
TARGET_ATR_MULT = 3.0
HARD_EXIT = dtime(15, 0)

# Sizing: matches the LIVE script's current formula -- see module docstring.
ALLOCATED_CAPITAL = 250_000
POSITION_PCT = 0.10
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT = 6

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_resized")
os.makedirs(OUT_DIR, exist_ok=True)

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


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


print(f"Universe: {len(UNIVERSE)} names")
all_trades = []
skipped = {}

for symbol in UNIVERSE:
    df15, err15 = _fetch(symbol, "15m")
    if err15:
        skipped[symbol] = err15
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    df30 = df15.resample("30min", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])

    df30["ema200"] = df30["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    df30["regime"] = np.where(df30["close"] > df30["ema200"], "BULL", "BEAR")
    regime_series = df30["regime"].copy()
    regime_series.index = df30.index

    df15["ema_fast"] = df15["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df15["ema_slow"] = df15["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df15["atr"] = compute_atr(df15)
    df15["bull_cross"] = (df15["ema_fast"].shift(1) <= df15["ema_slow"].shift(1)) & (df15["ema_fast"] > df15["ema_slow"])
    df15["bear_cross"] = (df15["ema_fast"].shift(1) >= df15["ema_slow"].shift(1)) & (df15["ema_fast"] < df15["ema_slow"])

    df15["regime"] = regime_series.reindex(df15.index, method="ffill")

    print(f"  {symbol}: {len(df15):,} 15m bars, {len(df30):,} 30m bars "
          f"({df15.index[0].date()} -> {df15.index[-1].date()})")

    df15 = df15.dropna(subset=["atr", "regime"])
    for i in range(len(df15) - 1):
        row = df15.iloc[i]
        direction = None
        if row["bull_cross"] and row["regime"] == "BULL":
            direction = "LONG"
        elif row["bear_cross"] and row["regime"] == "BEAR":
            direction = "SHORT"
        if direction is None:
            continue

        entry_ts = df15.index[i + 1]
        if entry_ts.date() != df15.index[i].date():
            continue
        entry_bar = df15.iloc[i + 1]
        entry_price = entry_bar["open"]
        atr = row["atr"]
        if direction == "LONG":
            stop_px = entry_price - STOP_ATR_MULT * atr
            target_px = entry_price + TARGET_ATR_MULT * atr
        else:
            stop_px = entry_price + STOP_ATR_MULT * atr
            target_px = entry_price - TARGET_ATR_MULT * atr

        exit_px, exit_ts, reason = None, None, None
        rest = df15[df15.index > entry_ts]
        for ts2, bar2 in rest.iterrows():
            t = ts2.time()
            if direction == "LONG":
                if bar2["low"] <= stop_px:
                    exit_px, exit_ts, reason = stop_px, ts2, "STOP"
                    break
                if bar2["high"] >= target_px:
                    exit_px, exit_ts, reason = target_px, ts2, "TARGET"
                    break
                if bar2["bear_cross"]:
                    exit_px, exit_ts, reason = bar2["close"], ts2, "REVERSE_CROSS"
                    break
            else:
                if bar2["high"] >= stop_px:
                    exit_px, exit_ts, reason = stop_px, ts2, "STOP"
                    break
                if bar2["low"] <= target_px:
                    exit_px, exit_ts, reason = target_px, ts2, "TARGET"
                    break
                if bar2["bull_cross"]:
                    exit_px, exit_ts, reason = bar2["close"], ts2, "REVERSE_CROSS"
                    break
            if t >= HARD_EXIT:
                exit_px, exit_ts, reason = bar2["close"], ts2, "HARD_EXIT"
                break
        if exit_px is None:
            continue

        pnl_pct = (exit_px - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_px) / entry_price
        all_trades.append({
            "symbol": symbol, "direction": direction, "entry_time": entry_ts,
            "entry_price": round(entry_price, 2), "stop_px": round(stop_px, 2),
            "target_px": round(target_px, 2), "exit_time": exit_ts,
            "exit_price": round(exit_px, 2), "pnl_pct": round(pnl_pct * 100, 3),
            "reason": reason,
        })

if not all_trades:
    print(f"\nNo trades generated at all. Skipped symbols: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nTotal signals across universe (before position-cap filtering): {len(trades_df)}")

# ---- Portfolio-level simulation: current live sizing formula, max 6 concurrent positions ----
capital_per_trade = ALLOCATED_CAPITAL * POSITION_PCT
buying_power = capital_per_trade * ASSUMED_MIS_LEVERAGE
open_positions = []
accepted = []
rejected_capacity = 0

for _, trade in trades_df.iterrows():
    open_positions = [p for p in open_positions if p > trade["entry_time"]]
    if len(open_positions) >= MAX_CONCURRENT:
        rejected_capacity += 1
        continue
    open_positions.append(trade["exit_time"])
    qty = int(buying_power // trade["entry_price"])
    pnl_rupees = qty * trade["entry_price"] * (trade["pnl_pct"] / 100)
    t = trade.to_dict()
    t["qty"] = qty
    t["pnl_rupees"] = round(pnl_rupees, 2)
    accepted.append(t)

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_resized_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected for exceeding {MAX_CONCURRENT}-position cap: {rejected_capacity}")

if result_df.empty:
    print("\nNo trades survived position-cap filtering.")
else:
    n = len(result_df)
    wins = (result_df["pnl_pct"] > 0).sum()
    wr = wins / n * 100
    avg_pct = result_df["pnl_pct"].mean()
    total_rupees = result_df["pnl_rupees"].sum()
    gw = result_df[result_df["pnl_pct"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_pct"] <= 0]["pnl_rupees"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n=== EMA Regime Crossover Backtest (resized sizing) ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L: {avg_pct:+.3f}% | PF: {pf:.2f}")
    print(f"Total P&L: Rs {total_rupees:+,.0f} on Rs {ALLOCATED_CAPITAL:,.0f} allocated capital "
          f"(Rs {buying_power:,.0f} buying power/trade)")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")

    def _pf(g):
        w = g.loc[g.pnl_rupees > 0, "pnl_rupees"].sum()
        losses = -g.loc[g.pnl_rupees < 0, "pnl_rupees"].sum()
        return w / losses if losses > 0 else float("inf")

    print("\nBy direction:")
    for d, g in result_df.groupby("direction"):
        print(f"  {d}: n={len(g)}, WR={((g.pnl_pct>0).mean()*100):.1f}%, "
              f"total_pnl={g.pnl_rupees.sum():+,.0f}, PF={_pf(g):.2f}")

    print(f"\nPer-symbol trade counts:")
    print(result_df["symbol"].value_counts().to_string())
