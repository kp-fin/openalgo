"""
EMA Regime Crossover -- Trailing-Target Partial-Exit Backtest (A/B test)
20-name universe (top 10 Nifty 50 + top 10 Nifty Next 50, static current
constituents, see equities-system/strategies/ema_regime_crossover.md) |
2021-07-01 -> 2026-06-30

Isolated ablation against the existing baseline (ema_regime_crossover_backtest.py):
only the exit mechanic changes. Regime (EMA200/30m), entry (EMA9/20 crossover
on 15m), original fixed stop (1.36xATR) / fixed target (3xATR) / reverse-cross /
15:00 hard exit, universe, window, and max-6-concurrent position cap are all
held identical to the baseline.

Sizing note: the original 2026-07-17 baseline backtest (and the 2026-07-20 RRG
A/B test) used a flat Rs 1 Cr / 12.5% "OpenAlgo Sandbox convention" for internal
backtest-to-backtest comparability, deliberately decoupled from the live script's
real-money sizing. This script instead uses the LIVE script's current, real
sizing formula (ema_regime_crossover_signal.py, corrected 2026-07-21) --
allocated_capital (Rs 2,50,000, paper mode) x 10% pre-leverage slice x 5x
ASSUMED_MIS_LEVERAGE -- since the much smaller per-trade notional changes
integer-share rounding enough to matter. A matching resized-baseline run
(ema_regime_crossover_backtest_resized.py) was produced alongside for a valid
apples-to-apples comparison at this same sizing, since the original baseline
CSV is not comparable at this different capital scale.

New rule (Karan-specified, 2026-07-28), layered on top of the unmodified
original exit logic:
  - Arm threshold: once favourable excursion from entry reaches 1.5xATR(14)
    (half the original fixed target), a trailing level activates.
  - Trail distance: 1xATR(14) behind the running favourable extreme (highest
    high since entry for LONG, lowest low for SHORT), recomputed every bar
    once armed.
  - First time price pulls back and touches the trailing level: exit 50% of
    qty at that level, reason="TRAIL_TARGET".
  - The remaining 50% reverts to the ORIGINAL, unmodified exit rule only --
    fixed stop, fixed target, reverse-cross, hard exit -- no further trailing
    checks apply to it (only the first trailing touch matters).
  - If the trade never arms (never reaches 1.5xATR profit) before some other
    exit fires, no partial exit ever happens -- 100% qty exits exactly as the
    baseline would.

Same-bar approximation (documented, not hidden): within a bar, the running
favourable extreme is updated using that bar's own high/low BEFORE checking
whether that same bar's opposite-side price touches the newly-updated trail
level -- the same "use this bar's own OHLC, no lookahead into future bars"
convention the baseline already uses for its stop/target checks. Ordering
within a bar (before any partial exit has happened): original STOP, then
original TARGET, then trail-touch (partial), then REVERSE_CROSS, then
HARD_EXIT-by-time -- so a bar that would hit the original stop or target
takes priority over a same-bar trail touch, consistent with the baseline's
own "stop checked before target" worst-case convention.

Each original signal now produces 1 output row (no partial ever happened,
qty_frac=1.0, reason as in baseline) or 2 rows (TRAIL_TARGET leg + a second
leg for the remaining qty, reason one of STOP/TARGET/REVERSE_CROSS/HARD_EXIT).
Qty is split so the two legs' quantities always sum exactly to the trade's
total qty (round-then-adjust-last-leg, avoids rounding leakage). The
portfolio-level 6-concurrent-position cap simulation treats a trade as "open"
until its LAST leg's exit time, same semantics as the baseline's single
exit_time.
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

# Sizing: matches the LIVE script's current formula (ema_regime_crossover_signal.py,
# corrected 2026-07-21 "capital-slice-then-leverage"), NOT the old flat-Rs-1-Cr/12.5%
# convention the original 2026-07-17 baseline backtest used. allocated_capital read from
# the live state file (strategies/scripts/state/capital_allocation.json) at the time this
# script was written -- Rs 2,50,000, paper mode. capital_per_trade is taken BEFORE
# leverage; the full leveraged buying_power is used for qty, no further cap layered on.
ALLOCATED_CAPITAL = 250_000
POSITION_PCT = 0.10
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT = 6

# New: trailing-target partial-exit parameters (Karan-confirmed 2026-07-28)
TRAIL_ARM_MULT = 2.0   # favourable excursion (x ATR) needed to arm the trail
TRAIL_ATR_MULT = 2.0   # trail distance (x ATR) behind the running favourable extreme
PARTIAL_FRACTION = 0.5  # fraction of qty exited at first trail touch

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_trailing_target")
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


def simulate_exit(direction, entry_price, atr, stop_px, target_px, rest_df):
    """Returns a list of 1-2 leg dicts: {reason, exit_time, exit_price, qty_frac}."""
    legs = []
    running_extreme = entry_price
    armed = False
    partial_done = False

    for ts2, bar2 in rest_df.iterrows():
        t = ts2.time()
        if not partial_done:
            if direction == "LONG":
                if bar2["low"] <= stop_px:
                    legs.append({"reason": "STOP", "exit_time": ts2, "exit_price": stop_px, "qty_frac": 1.0})
                    return legs
                if bar2["high"] >= target_px:
                    legs.append({"reason": "TARGET", "exit_time": ts2, "exit_price": target_px, "qty_frac": 1.0})
                    return legs
                running_extreme = max(running_extreme, bar2["high"])
                if not armed and (running_extreme - entry_price) >= TRAIL_ARM_MULT * atr:
                    armed = True
                if armed:
                    trail_level = running_extreme - TRAIL_ATR_MULT * atr
                    if bar2["low"] <= trail_level:
                        legs.append({"reason": "TRAIL_TARGET", "exit_time": ts2, "exit_price": trail_level,
                                     "qty_frac": PARTIAL_FRACTION})
                        partial_done = True
                        continue
                if bar2["bear_cross"]:
                    legs.append({"reason": "REVERSE_CROSS", "exit_time": ts2, "exit_price": bar2["close"], "qty_frac": 1.0})
                    return legs
            else:  # SHORT
                if bar2["high"] >= stop_px:
                    legs.append({"reason": "STOP", "exit_time": ts2, "exit_price": stop_px, "qty_frac": 1.0})
                    return legs
                if bar2["low"] <= target_px:
                    legs.append({"reason": "TARGET", "exit_time": ts2, "exit_price": target_px, "qty_frac": 1.0})
                    return legs
                running_extreme = min(running_extreme, bar2["low"])
                if not armed and (entry_price - running_extreme) >= TRAIL_ARM_MULT * atr:
                    armed = True
                if armed:
                    trail_level = running_extreme + TRAIL_ATR_MULT * atr
                    if bar2["high"] >= trail_level:
                        legs.append({"reason": "TRAIL_TARGET", "exit_time": ts2, "exit_price": trail_level,
                                     "qty_frac": PARTIAL_FRACTION})
                        partial_done = True
                        continue
                if bar2["bull_cross"]:
                    legs.append({"reason": "REVERSE_CROSS", "exit_time": ts2, "exit_price": bar2["close"], "qty_frac": 1.0})
                    return legs
            if t >= HARD_EXIT:
                legs.append({"reason": "HARD_EXIT", "exit_time": ts2, "exit_price": bar2["close"], "qty_frac": 1.0})
                return legs
        else:
            remaining_frac = 1.0 - PARTIAL_FRACTION
            if direction == "LONG":
                if bar2["low"] <= stop_px:
                    legs.append({"reason": "STOP", "exit_time": ts2, "exit_price": stop_px, "qty_frac": remaining_frac})
                    return legs
                if bar2["high"] >= target_px:
                    legs.append({"reason": "TARGET", "exit_time": ts2, "exit_price": target_px, "qty_frac": remaining_frac})
                    return legs
                if bar2["bear_cross"]:
                    legs.append({"reason": "REVERSE_CROSS", "exit_time": ts2, "exit_price": bar2["close"],
                                 "qty_frac": remaining_frac})
                    return legs
            else:
                if bar2["high"] >= stop_px:
                    legs.append({"reason": "STOP", "exit_time": ts2, "exit_price": stop_px, "qty_frac": remaining_frac})
                    return legs
                if bar2["low"] <= target_px:
                    legs.append({"reason": "TARGET", "exit_time": ts2, "exit_price": target_px, "qty_frac": remaining_frac})
                    return legs
                if bar2["bull_cross"]:
                    legs.append({"reason": "REVERSE_CROSS", "exit_time": ts2, "exit_price": bar2["close"],
                                 "qty_frac": remaining_frac})
                    return legs
            if t >= HARD_EXIT:
                legs.append({"reason": "HARD_EXIT", "exit_time": ts2, "exit_price": bar2["close"],
                             "qty_frac": remaining_frac})
                return legs
    return legs


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

        rest = df15[df15.index > entry_ts]
        legs = simulate_exit(direction, entry_price, atr, stop_px, target_px, rest)
        if not legs:
            continue

        all_trades.append({
            "symbol": symbol, "direction": direction, "entry_time": entry_ts,
            "entry_price": entry_price, "stop_px": stop_px, "target_px": target_px,
            "legs": legs, "final_exit_time": legs[-1]["exit_time"],
        })

if not all_trades:
    print(f"\nNo trades generated at all. Skipped symbols: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nTotal signals across universe (before position-cap filtering): {len(trades_df)}")

# ---- Portfolio-level simulation: current live sizing formula, max 6 concurrent positions ----
# A trade occupies a slot until its LAST leg's exit time (matches baseline semantics,
# generalised from a single exit_time to final_exit_time).
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
    open_positions.append(trade["final_exit_time"])
    accepted.append(trade)

output_rows = []
for trade_id, trade in enumerate(accepted):
    qty = int(buying_power // trade["entry_price"])
    legs = trade["legs"]
    qtys = [round(qty * leg["qty_frac"]) for leg in legs]
    if len(qtys) > 1:
        qtys[-1] = qty - sum(qtys[:-1])  # conserve total qty exactly, absorb rounding in last leg

    for leg_num, (leg, leg_qty) in enumerate(zip(legs, qtys), start=1):
        pnl_pct = ((leg["exit_price"] - trade["entry_price"]) / trade["entry_price"] if trade["direction"] == "LONG"
                   else (trade["entry_price"] - leg["exit_price"]) / trade["entry_price"])
        pnl_rupees = leg_qty * trade["entry_price"] * pnl_pct
        output_rows.append({
            "trade_id": trade_id, "leg": leg_num, "symbol": trade["symbol"],
            "direction": trade["direction"], "entry_time": trade["entry_time"],
            "entry_price": round(trade["entry_price"], 2), "stop_px": round(trade["stop_px"], 2),
            "target_px": round(trade["target_px"], 2), "exit_time": leg["exit_time"],
            "exit_price": round(leg["exit_price"], 2), "pnl_pct": round(pnl_pct * 100, 3),
            "reason": leg["reason"], "qty": leg_qty, "pnl_rupees": round(pnl_rupees, 2),
        })

result_df = pd.DataFrame(output_rows)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_trailing_target_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected for exceeding {MAX_CONCURRENT}-position cap: {rejected_capacity}")

if result_df.empty:
    print("\nNo trades survived position-cap filtering.")
else:
    n_trades = len(accepted)
    n_legs = len(result_df)
    partial_trades = (result_df["reason"] == "TRAIL_TARGET").sum()
    wins = (result_df["pnl_rupees"] > 0).sum()
    wr = wins / n_legs * 100
    total_rupees = result_df["pnl_rupees"].sum()
    gw = result_df[result_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    print(f"\n=== EMA Regime Crossover -- Trailing-Target Backtest ===")
    print(f"Original trades: {n_trades} | Output legs: {n_legs} | Trades with a partial exit: {partial_trades} "
          f"({partial_trades/n_trades*100:.1f}%)")
    print(f"Leg-level WR: {wr:.1f}% | PF: {pf:.2f}")
    print(f"Total P&L: Rs {total_rupees:+,.0f} on Rs {ALLOCATED_CAPITAL:,.0f} allocated capital "
          f"(Rs {buying_power:,.0f} buying power/trade)")
    print(f"Exit breakdown (legs): {result_df['reason'].value_counts().to_dict()}")

    def _pf(g):
        w = g.loc[g.pnl_rupees > 0, "pnl_rupees"].sum()
        losses = -g.loc[g.pnl_rupees < 0, "pnl_rupees"].sum()
        return w / losses if losses > 0 else float("inf")

    print("\nBy direction:")
    for d, g in result_df.groupby("direction"):
        print(f"  {d}: legs={len(g)}, WR={((g.pnl_rupees>0).mean()*100):.1f}%, "
              f"total_pnl={g.pnl_rupees.sum():+,.0f}, PF={_pf(g):.2f}")

    print("\nPer-symbol total P&L:")
    print(result_df.groupby("symbol")["pnl_rupees"].sum().sort_values(ascending=False).round(0).to_string())
