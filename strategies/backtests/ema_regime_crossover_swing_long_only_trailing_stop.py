"""
EMA Regime Crossover — SWING, LONG-ONLY — trailing-stop variant

Locked-in config from prior sweeps: RS_MOMENTUM_SLOPE_LOOKBACK=2, initial
stop=3.25xATR (see ema_regime_crossover_swing_long_only_stop_sweep_target8.py:
WR 40.5%, PF 1.77, +74.08L, 12/19 profitable, 257 trades, target=8.0xATR
fixed). This script REPLACES the fixed 8xATR target with a chandelier-style
trailing stop, to test whether letting winners run past 8xATR (with a
ratcheting stop instead of a hard ceiling) beats the fixed-target version.
Karan's request 2026-07-19.

Trailing mechanic:
  trail_stop_t = highest_high_since_entry(through bar t-1) - TRAIL_ATR_MULT * ATR_t
  effective_stop_t = max(initial_stop, trail_stop_t)   -- ratchets up only, never down
ATR is recomputed each day (not frozen at entry) since a 20-30 day hold can
span a real volatility-regime change. TRAIL_ATR_MULT locked at 6.0 (see
ema_regime_crossover_swing_long_only_trail_mult_sweep.py -- 3.25 was tested
first and underperformed the fixed 8xATR target; PF/total P&L climb sharply
from 3.25->6.0 then plateau, so 6.0 is the practical sweet spot, not 8.0/10.0
which add almost nothing more).

Exit: trailing-stop hit (label TRAIL_STOP) or opposite EMA9/EMA20 crossover
(REVERSE_CROSS), whichever comes first. No fixed profit target anymore.
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

ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14
TOTAL_CAPITAL = 10_000_000
POSITION_PCT = 0.125
MAX_CONCURRENT = 6

RS_ZSCORE_WINDOW = 10
RS_SMOOTH_WINDOW = 2
RS_MOMENTUM_SLOPE_LOOKBACK = 2   # locked

INITIAL_STOP_ATR_MULT = 3.25     # locked
TRAIL_ATR_MULT = 6.0             # locked 2026-07-19 -- see trail_mult sweep (PF plateaus from 6.0 onward)

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10
INDEX_FOR = {**{s: "NIFTY" for s in NIFTY50_TOP10}, **{s: "NIFTYNXT50" for s in NEXT50_TOP10}}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_swing_long_only_trailing_stop")
os.makedirs(OUT_DIR, exist_ok=True)

from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)


def _fetch_daily(symbol, exchange):
    try:
        resp = client.history(symbol=symbol, exchange=exchange, interval="D",
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
    df.index = df.index.normalize()
    return df.sort_index(), None


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_rs_ratio_momentum(stock_close, index_close):
    rs = 100 * (stock_close / index_close)
    rs_ratio_raw = 100 + (rs - rs.rolling(RS_ZSCORE_WINDOW).mean()) / rs.rolling(RS_ZSCORE_WINDOW).std()
    rs_ratio = rs_ratio_raw.rolling(RS_SMOOTH_WINDOW).mean()
    mom_raw = 100 + (rs_ratio - rs_ratio.rolling(RS_ZSCORE_WINDOW).mean()) / rs_ratio.rolling(RS_ZSCORE_WINDOW).std()
    rs_momentum = mom_raw.rolling(RS_SMOOTH_WINDOW).mean()
    return rs_ratio, rs_momentum


index_data = {}
for idx_symbol in ("NIFTY", "NIFTYNXT50"):
    df_idx, err = _fetch_daily(idx_symbol, "NSE_INDEX")
    if err:
        raise SystemExit(f"Could not fetch index {idx_symbol}: {err}")
    index_data[idx_symbol] = df_idx

print(f"Universe: {len(UNIVERSE)} names (LONG-ONLY, LOOKBACK=2, TRAILING STOP {TRAIL_ATR_MULT}xATR)")
all_trades = []
skipped = {}

for symbol in UNIVERSE:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    idx_df = index_data[INDEX_FOR[symbol]]
    combined_close = pd.DataFrame({"stock": df["close"], "index": idx_df["close"]}).dropna()
    rs_ratio, rs_momentum = compute_rs_ratio_momentum(combined_close["stock"], combined_close["index"])

    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])
    df["rs_ratio"] = rs_ratio.reindex(df.index)
    df["rs_momentum"] = rs_momentum.reindex(df.index)
    rs_momentum_rising = df["rs_momentum"] > df["rs_momentum"].shift(RS_MOMENTUM_SLOPE_LOOKBACK)
    bull_regime = (df["rs_ratio"] > 100) & ((df["rs_momentum"] > 100) | rs_momentum_rising)
    df["regime"] = np.where(bull_regime, "BULL", "NOT_BULL")

    print(f"  {symbol} (vs {INDEX_FOR[symbol]}): {len(df):,} usable daily bars "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    df = df.dropna(subset=["atr", "rs_ratio", "rs_momentum"])
    n = len(df)
    for i in range(n - 1):
        row = df.iloc[i]
        if not (row["bull_cross"] and row["regime"] == "BULL"):
            continue

        entry_ts = df.index[i + 1]
        entry_bar = df.iloc[i + 1]
        entry_price = entry_bar["open"]
        atr_entry = row["atr"]
        initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr_entry

        trail_stop = initial_stop
        highest_high = entry_price
        exit_px, exit_ts, reason = None, None, None
        rest = df[df.index > entry_ts]
        for ts2, bar2 in rest.iterrows():
            if bar2["low"] <= trail_stop:
                exit_px, exit_ts, reason = trail_stop, ts2, "TRAIL_STOP"
                break
            if bar2["bear_cross"]:
                exit_px, exit_ts, reason = bar2["close"], ts2, "REVERSE_CROSS"
                break
            # Update the trailing level AFTER today's check, using today's high/ATR --
            # ratchets up only, takes effect from tomorrow's bar (no same-bar lookahead).
            highest_high = max(highest_high, bar2["high"])
            candidate_trail = highest_high - TRAIL_ATR_MULT * bar2["atr"]
            trail_stop = max(trail_stop, candidate_trail)
        if exit_px is None:
            if len(rest) == 0:
                continue
            last_ts, last_bar = rest.index[-1], rest.iloc[-1]
            exit_px, exit_ts, reason = last_bar["close"], last_ts, "END_OF_DATA"

        pnl_pct = (exit_px - entry_price) / entry_price
        all_trades.append({
            "symbol": symbol, "entry_time": entry_ts, "entry_price": round(entry_price, 2),
            "initial_stop": round(initial_stop, 2), "exit_time": exit_ts,
            "exit_price": round(exit_px, 2), "pnl_pct": round(pnl_pct * 100, 3),
            "reason": reason, "holding_days": (exit_ts - entry_ts).days,
        })

if not all_trades:
    print(f"\nNo trades generated at all. Skipped symbols: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nTotal signals across universe (before position-cap filtering): {len(trades_df)}")

position_size = TOTAL_CAPITAL * POSITION_PCT
open_positions = []
accepted = []
rejected_capacity = 0
for _, trade in trades_df.iterrows():
    open_positions = [p for p in open_positions if p > trade["entry_time"]]
    if len(open_positions) >= MAX_CONCURRENT:
        rejected_capacity += 1
        continue
    open_positions.append(trade["exit_time"])
    qty = int(position_size // trade["entry_price"])
    pnl_rupees = qty * trade["entry_price"] * (trade["pnl_pct"] / 100)
    t = trade.to_dict()
    t["qty"] = qty
    t["pnl_rupees"] = round(pnl_rupees, 2)
    accepted.append(t)

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_swing_long_only_trailing_stop_trades.csv"), index=False)

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
    avg_holding = result_df["holding_days"].mean()
    by_symbol = result_df.groupby("symbol")["pnl_rupees"].sum()
    profitable_symbols = (by_symbol > 0).sum()
    print(f"\n=== EMA Regime Crossover SWING Backtest (LONG-ONLY, TRAILING STOP) ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L: {avg_pct:+.3f}% | PF: {pf:.2f}")
    print(f"Total P&L: Rs {total_rupees:+,.0f} on Rs {TOTAL_CAPITAL:,.0f} capital")
    print(f"Avg holding period: {avg_holding:.1f} calendar days")
    print(f"Profitable symbols: {profitable_symbols}/{len(by_symbol)}")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"\nPer-symbol P&L:")
    print(by_symbol.sort_values().to_string())
