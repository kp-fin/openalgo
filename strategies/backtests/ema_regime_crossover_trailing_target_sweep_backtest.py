"""
EMA Regime Crossover -- Trailing-Target Arm/Trail Parameter Sweep

Follow-up to ema_regime_crossover_trailing_target_backtest.py, which tested one
fixed combination (arm=1.5xATR, trail=1.0xATR, 50% partial exit) and found it
made the strategy worse across the board (combined PF 1.06 -> 0.97, both LONG
and SHORT PF worsened) -- the mechanism fired on ~30% of trades, clipping the
strategy's two best-performing exit types (TARGET, HARD_EXIT) short before
their "let it run" behaviour played out. This sweeps the two free parameters
to check whether a looser combination avoids that problem, or whether the
whole approach is a dead end regardless of tuning.

Grid: arm in {1.0, 1.5, 2.0, 2.5} x ATR, trail in {0.5, 1.0, 1.5, 2.0} x ATR
(16 combinations). arm=3.0 (== the fixed target multiple) would be degenerate
-- the full-target check has priority over the trail check in the same bar
(matching the single-test script's ordering), so arming only once price has
already reached the full target means the trail could never fire first --
excluded from the grid for that reason. A synthetic NO_TRAIL row (trailing
disabled entirely) is included as an in-table reference point, and should
exactly reproduce ema_regime_crossover_backtest_resized.py's numbers (PF 1.06
combined) as a correctness check on this script's own baseline path.

Sizing: same corrected current-live formula as the single-test script and its
resized baseline -- allocated_capital (Rs 2,50,000, paper mode) x 10%
pre-leverage slice x 5x ASSUMED_MIS_LEVERAGE, NOT the original 2026-07-17
backtest's flat Rs 1 Cr/12.5% convention.

Efficiency note: the network fetch + indicator computation (the expensive
part) runs ONCE per symbol, cached in memory; only the cheap, pure-Python
exit-simulation + portfolio-cap-simulation re-runs per grid combination. The
portfolio-cap simulation (max 6 concurrent) is re-run per combination, not
just the per-trade P&L -- different arm/trail settings change exit timing,
which changes which trades the 6-position cap accepts or rejects.
"""

import os
import warnings
from datetime import time as dtime
from itertools import product

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
CAPITAL_PER_TRADE = ALLOCATED_CAPITAL * POSITION_PCT
BUYING_POWER = CAPITAL_PER_TRADE * ASSUMED_MIS_LEVERAGE

PARTIAL_FRACTION = 0.5
ARM_GRID = [1.0, 1.5, 2.0, 2.5]
TRAIL_GRID = [0.5, 1.0, 1.5, 2.0]

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_trailing_target_sweep")
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


def simulate_exit(direction, entry_price, atr, stop_px, target_px, rest_df, arm_mult, trail_mult):
    """arm_mult=None disables trailing entirely (NO_TRAIL reference row)."""
    legs = []
    running_extreme = entry_price
    armed = arm_mult is None  # if disabled, treat as "never arms" via early-outs below
    partial_done = False
    trailing_enabled = arm_mult is not None

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
                if trailing_enabled:
                    running_extreme = max(running_extreme, bar2["high"])
                    if not armed and (running_extreme - entry_price) >= arm_mult * atr:
                        armed = True
                    if armed:
                        trail_level = running_extreme - trail_mult * atr
                        if bar2["low"] <= trail_level:
                            legs.append({"reason": "TRAIL_TARGET", "exit_time": ts2, "exit_price": trail_level,
                                         "qty_frac": PARTIAL_FRACTION})
                            partial_done = True
                            continue
                if bar2["bear_cross"]:
                    legs.append({"reason": "REVERSE_CROSS", "exit_time": ts2, "exit_price": bar2["close"], "qty_frac": 1.0})
                    return legs
            else:
                if bar2["high"] >= stop_px:
                    legs.append({"reason": "STOP", "exit_time": ts2, "exit_price": stop_px, "qty_frac": 1.0})
                    return legs
                if bar2["low"] <= target_px:
                    legs.append({"reason": "TARGET", "exit_time": ts2, "exit_price": target_px, "qty_frac": 1.0})
                    return legs
                if trailing_enabled:
                    running_extreme = min(running_extreme, bar2["low"])
                    if not armed and (entry_price - running_extreme) >= arm_mult * atr:
                        armed = True
                    if armed:
                        trail_level = running_extreme + trail_mult * atr
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


# ---- Fetch + prep once ----
print(f"Universe: {len(UNIVERSE)} names -- fetching once, sweeping {len(ARM_GRID)*len(TRAIL_GRID)+1} "
      f"parameter combinations on the cached data\n")

signals = []  # each: symbol, direction, entry_time, entry_price, atr, stop_px, target_px
prepared = {}  # symbol -> df15 (indexed, with atr/regime/cross columns)
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
    df15 = df15.dropna(subset=["atr", "regime"])
    prepared[symbol] = df15
    print(f"  {symbol}: {len(df15):,} 15m bars ({df15.index[0].date()} -> {df15.index[-1].date()})")

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
        signals.append({"symbol": symbol, "direction": direction, "entry_time": entry_ts,
                         "entry_price": entry_price, "atr": atr, "stop_px": stop_px, "target_px": target_px})

signals.sort(key=lambda s: s["entry_time"])
print(f"\nTotal signal candidates: {len(signals)} (data fetch/prep done once)\n")


def _pf(pnl_series):
    w = pnl_series[pnl_series > 0].sum()
    l = -pnl_series[pnl_series <= 0].sum()
    return w / l if l > 0 else float("inf")


def run_combo(arm_mult, trail_mult, label):
    all_trades = []
    for sig in signals:
        df15 = prepared[sig["symbol"]]
        rest = df15[df15.index > sig["entry_time"]]
        legs = simulate_exit(sig["direction"], sig["entry_price"], sig["atr"], sig["stop_px"], sig["target_px"],
                              rest, arm_mult, trail_mult)
        if not legs:
            continue
        all_trades.append({**sig, "legs": legs, "final_exit_time": legs[-1]["exit_time"]})

    open_positions = []
    accepted = []
    for trade in all_trades:
        open_positions = [p for p in open_positions if p > trade["entry_time"]]
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        open_positions.append(trade["final_exit_time"])
        accepted.append(trade)

    rows = []
    for trade in accepted:
        qty = int(BUYING_POWER // trade["entry_price"])
        legs = trade["legs"]
        qtys = [round(qty * leg["qty_frac"]) for leg in legs]
        if len(qtys) > 1:
            qtys[-1] = qty - sum(qtys[:-1])
        for leg, leg_qty in zip(legs, qtys):
            pnl_pct = ((leg["exit_price"] - trade["entry_price"]) / trade["entry_price"] if trade["direction"] == "LONG"
                       else (trade["entry_price"] - leg["exit_price"]) / trade["entry_price"])
            rows.append({"symbol": trade["symbol"], "direction": trade["direction"], "reason": leg["reason"],
                         "qty": leg_qty, "pnl_rupees": leg_qty * trade["entry_price"] * pnl_pct})

    df = pd.DataFrame(rows)
    n_trades = len(accepted)
    partial_n = sum(1 for t in accepted if len(t["legs"]) > 1)
    if df.empty:
        return {"combo": label, "arm": arm_mult, "trail": trail_mult, "n_trades": n_trades,
                "partial_pct": 0, "wr": 0, "pf": 0, "total_pnl": 0, "long_pf": 0, "short_pf": 0}
    long_df = df[df.direction == "LONG"]["pnl_rupees"]
    short_df = df[df.direction == "SHORT"]["pnl_rupees"]
    return {
        "combo": label, "arm": arm_mult, "trail": trail_mult, "n_trades": n_trades,
        "partial_pct": round(partial_n / n_trades * 100, 1) if n_trades else 0,
        "wr": round((df.pnl_rupees > 0).mean() * 100, 1),
        "pf": round(_pf(df.pnl_rupees), 3),
        "total_pnl": round(df.pnl_rupees.sum(), 0),
        "long_pf": round(_pf(long_df), 3) if len(long_df) else 0,
        "short_pf": round(_pf(short_df), 3) if len(short_df) else 0,
    }


results = [run_combo(None, None, "NO_TRAIL (reference baseline)")]
for arm_mult, trail_mult in product(ARM_GRID, TRAIL_GRID):
    label = f"arm={arm_mult}x / trail={trail_mult}x"
    print(f"Running {label} ...")
    results.append(run_combo(arm_mult, trail_mult, label))

sweep_df = pd.DataFrame(results).sort_values("pf", ascending=False).reset_index(drop=True)
sweep_df.to_csv(os.path.join(OUT_DIR, "sweep_results.csv"), index=False)

print(f"\n{'='*100}")
print("SWEEP RESULTS (sorted by combined PF, best first)")
print(f"{'='*100}")
print(sweep_df.to_string(index=False))

best = sweep_df.iloc[0]
baseline_row = sweep_df[sweep_df["combo"] == "NO_TRAIL (reference baseline)"].iloc[0]
print(f"\nBest combo: {best['combo']} -- PF {best['pf']}, total P&L Rs {best['total_pnl']:+,.0f}")
print(f"Reference (no trail): PF {baseline_row['pf']}, total P&L Rs {baseline_row['total_pnl']:+,.0f}")
print(f"Combos beating the no-trail baseline's PF: {(sweep_df['pf'] > baseline_row['pf']).sum() - 1} of {len(ARM_GRID)*len(TRAIL_GRID)}")
