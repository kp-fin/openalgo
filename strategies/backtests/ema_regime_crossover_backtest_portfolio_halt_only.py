"""
EMA Regime Crossover -- Portfolio-Halt-ONLY Backtest (isolates what's actually live)

Found 2026-07-28: the per-symbol same-day loss-block (added to source
2026-07-27, commit e412c765) was NEVER copied to the deployed/scheduled
script (strategies/scripts/ema_regime_crossover_signal_20260717094335.py,
last modified 2026-07-21 -- six days before that commit). Confirmed via
today's live state file (ema_regime_crossover_state.json): only
daily_realized_pnl/daily_loss_halted are present, no daily_loss_blocked key
at all. So the live strategy has ONLY ever run the portfolio-wide 2% daily
halt -- the per-symbol block modeled in
ema_regime_crossover_backtest_daily_circuit_breaker.py (PF 1.13, drawdown
20.9%) was never actually live, and that combined figure overstates the
real historical protection.

This script isolates just the portfolio-wide halt (matches what was
genuinely running through 2026-07-28, before today's deploy fix) to get the
real historical drawdown/PF figure for the gap period. Same regime/entry/
original-exit mechanics, universe, window, sizing formula, max-6-concurrent
cap as the resized baseline and the combined-circuit-breaker backtest --
only the per-symbol block is removed here, isolating the portfolio halt's
standalone effect.

Implementation note: same event-driven chronological walk as the combined
circuit-breaker backtest (exits must be flushed before evaluating a new
candidate's entry, since daily_realized_pnl depends on which trades have
already closed) -- just without the blocked-symbol set.
"""

import heapq
import os
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

ALLOCATED_CAPITAL = 250_000
POSITION_PCT = 0.10
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT = 6
DAILY_LOSS_PCT = 0.02  # matches the live script's DAILY_LOSS_PCT exactly

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_portfolio_halt_only")
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

    print(f"  {symbol}: {len(df15):,} 15m bars ({df15.index[0].date()} -> {df15.index[-1].date()})")

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
            "entry_price": entry_price, "stop_px": stop_px, "target_px": target_px,
            "exit_time": exit_ts, "exit_price": exit_px, "pnl_pct": pnl_pct, "reason": reason,
        })

if not all_trades:
    print(f"\nNo trades generated at all. Skipped symbols: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nTotal signal candidates: {len(trades_df)}")

# ---- Event-driven portfolio simulation: concurrency cap + PORTFOLIO-WIDE halt only ----
capital_per_trade = ALLOCATED_CAPITAL * POSITION_PCT
buying_power = capital_per_trade * ASSUMED_MIS_LEVERAGE
daily_loss_limit = DAILY_LOSS_PCT * ALLOCATED_CAPITAL

pending_exits = []  # heap of (exit_time, seq, trade_dict_with_pnl_rupees)
seq = 0
open_count = 0
current_day = None
daily_realized_pnl = 0.0
daily_loss_halted = False

accepted = []
rejected_cap, rejected_halt = 0, 0

for _, cand in trades_df.iterrows():
    qty = int(buying_power // cand["entry_price"])
    pnl_rupees = qty * cand["entry_price"] * cand["pnl_pct"]

    # Flush all pending exits at or before this candidate's entry time.
    while pending_exits and pending_exits[0][0] <= cand["entry_time"]:
        ex_time, _, ex_trade = heapq.heappop(pending_exits)
        if ex_time.date() == current_day:
            daily_realized_pnl += ex_trade["pnl_rupees"]
        open_count -= 1

    # New day? Reset circuit-breaker state. (Every trade closes same-day, so
    # all of the previous day's exits are guaranteed already flushed above.)
    if cand["entry_time"].date() != current_day:
        current_day = cand["entry_time"].date()
        daily_realized_pnl = 0.0
        daily_loss_halted = False

    if daily_realized_pnl <= -daily_loss_limit:
        daily_loss_halted = True

    if daily_loss_halted:
        rejected_halt += 1
        continue
    if open_count >= MAX_CONCURRENT:
        rejected_cap += 1
        continue

    t = cand.to_dict()
    t["qty"] = qty
    t["pnl_rupees"] = round(pnl_rupees, 2)
    accepted.append(t)
    open_count += 1
    seq += 1
    heapq.heappush(pending_exits, (cand["exit_time"], seq, t))

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_portfolio_halt_only_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | daily circuit-breaker halt (portfolio-wide only): {rejected_halt}")

if result_df.empty:
    print("\nNo trades survived.")
else:
    n = len(result_df)
    wr = (result_df["pnl_rupees"] > 0).mean() * 100
    total_rupees = result_df["pnl_rupees"].sum()
    gw = result_df[result_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n=== EMA Regime Crossover -- Portfolio-Halt-ONLY Backtest (what's actually been live) ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | PF: {pf:.2f}")
    print(f"Total P&L: Rs {total_rupees:+,.0f} on Rs {ALLOCATED_CAPITAL:,.0f} allocated capital")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")

    def _pf(g):
        w = g.loc[g.pnl_rupees > 0, "pnl_rupees"].sum()
        losses = -g.loc[g.pnl_rupees < 0, "pnl_rupees"].sum()
        return w / losses if losses > 0 else float("inf")

    print("\nBy direction:")
    for d, g in result_df.groupby("direction"):
        print(f"  {d}: n={len(g)}, WR={((g.pnl_rupees>0).mean()*100):.1f}%, "
              f"total_pnl={g.pnl_rupees.sum():+,.0f}, PF={_pf(g):.2f}")

    # ---- Drawdown, directly comparable to the resized-baseline figure ----
    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl"] = dd_df["pnl_rupees"].cumsum()
    dd_df["running_peak"] = dd_df["cum_pnl"].cummax()
    dd_df["drawdown"] = dd_df["cum_pnl"] - dd_df["running_peak"]
    max_dd = dd_df["drawdown"].min()
    max_dd_idx = dd_df["drawdown"].idxmin()
    peak_before = dd_df.loc[:max_dd_idx, "cum_pnl"].idxmax()
    after = dd_df.loc[max_dd_idx:]
    recovery = after[after["cum_pnl"] >= dd_df.loc[peak_before, "cum_pnl"]]

    print(f"\n=== Drawdown ===")
    print(f"Max drawdown: Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")
    print(f"Drawdown window: {dd_df.loc[peak_before, 'exit_time']} -> {dd_df.loc[max_dd_idx, 'exit_time']}")
    if len(recovery):
        print(f"Recovered by: {recovery.iloc[0]['exit_time']}")
    else:
        print("Never recovered by end of backtest")

    streak = 0
    max_streak = 0
    for v in (dd_df["pnl_rupees"] < 0):
        streak = streak + 1 if v else 0
        max_streak = max(max_streak, streak)
    print(f"Max consecutive losing trades: {max_streak}")

    print(f"\nPer-symbol trade counts:")
    print(result_df["symbol"].value_counts().to_string())
