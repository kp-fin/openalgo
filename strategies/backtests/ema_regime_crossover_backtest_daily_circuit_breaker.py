"""
EMA Regime Crossover -- Daily-Loss Circuit Breaker Retrofit

The live script (ema_regime_crossover_signal.py) already runs two portfolio-
level risk controls that NO backtest so far has modeled:
  1. Daily loss circuit breaker: once today's cumulative REALIZED P&L drops to
     or below -2% of allocated_capital, halt all NEW entries for the rest of
     the day. Existing open positions are never force-closed -- they keep
     riding to their own exit (stop/target/reverse-cross/hard-exit).
  2. Per-symbol same-day re-entry block: any symbol with a losing exit today
     cannot be re-entered again that same day (independent of the portfolio-
     wide halt -- can trip even before the 2% breaker does).
Both reset at the start of each new trading day.

Motivation: the resized baseline (ema_regime_crossover_backtest_resized.py,
same exit logic, same current sizing) has a max drawdown of Rs 1,72,073 --
68.8% of the Rs 2,50,000 allocated capital -- taking from 2021-08-04 to
2023-12-14 to recover. That backtest never modeled either control above, so
its drawdown is likely an overestimate of what live trading would actually
see. This retrofits both rules and re-measures the SAME drawdown metric for
a direct before/after comparison.

Everything else (regime, entry, original exit mechanics, universe, window,
sizing formula, max-6-concurrent cap) is identical to the resized baseline --
this is an isolated addition of portfolio-level risk controls only.

Implementation note: the daily circuit breaker depends on which trades have
ALREADY CLOSED (and their P&L) at the moment a new candidate's entry is being
evaluated -- not just entry order. This requires a true chronological
event-driven walk (process every pending exit at or before the current
candidate's entry time -- updating today's realized P&L and blocked-symbol
set -- BEFORE deciding whether to accept that candidate), rather than the
simpler entry-time-ordered concurrency check the other backtests use. Since
every trade in this strategy is closed same-day (hard exit by 15:00, verified
safe earlier in this backtest family), "day" can be tracked from each event's
own date with no overnight-carry edge cases.
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_daily_circuit_breaker")
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

# ---- Event-driven portfolio simulation: concurrency cap + daily circuit breaker ----
capital_per_trade = ALLOCATED_CAPITAL * POSITION_PCT
buying_power = capital_per_trade * ASSUMED_MIS_LEVERAGE
daily_loss_limit = DAILY_LOSS_PCT * ALLOCATED_CAPITAL

pending_exits = []  # heap of (exit_time, seq, trade_dict_with_pnl_rupees)
seq = 0
open_count = 0
current_day = None
daily_realized_pnl = 0.0
daily_loss_halted = False
daily_blocked = set()

accepted = []
rejected_cap, rejected_halt, rejected_symbol_block = 0, 0, 0

for _, cand in trades_df.iterrows():
    qty = int(buying_power // cand["entry_price"])
    pnl_rupees = qty * cand["entry_price"] * cand["pnl_pct"]

    # Flush all pending exits at or before this candidate's entry time.
    while pending_exits and pending_exits[0][0] <= cand["entry_time"]:
        ex_time, _, ex_trade = heapq.heappop(pending_exits)
        if ex_time.date() == current_day:
            daily_realized_pnl += ex_trade["pnl_rupees"]
            if ex_trade["pnl_rupees"] < 0:
                daily_blocked.add(ex_trade["symbol"])
        open_count -= 1

    # New day? Reset circuit-breaker state. (Every trade closes same-day, so
    # all of the previous day's exits are guaranteed already flushed above.)
    if cand["entry_time"].date() != current_day:
        current_day = cand["entry_time"].date()
        daily_realized_pnl = 0.0
        daily_loss_halted = False
        daily_blocked = set()

    if daily_realized_pnl <= -daily_loss_limit:
        daily_loss_halted = True

    if daily_loss_halted:
        rejected_halt += 1
        continue
    if cand["symbol"] in daily_blocked:
        rejected_symbol_block += 1
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
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_daily_circuit_breaker_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | daily circuit-breaker halt: {rejected_halt} | "
      f"per-symbol same-day loss block: {rejected_symbol_block}")

if result_df.empty:
    print("\nNo trades survived.")
else:
    n = len(result_df)
    wr = (result_df["pnl_rupees"] > 0).mean() * 100
    total_rupees = result_df["pnl_rupees"].sum()
    gw = result_df[result_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n=== EMA Regime Crossover -- Daily Circuit Breaker Backtest ===")
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
