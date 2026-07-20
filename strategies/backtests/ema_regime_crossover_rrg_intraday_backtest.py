"""
EMA Regime Crossover — RRG Regime A/B Test (intraday mechanics, RRG regime source)

Isolated A/B test against strategies/backtests/ema_regime_crossover_backtest.py.
Every mechanic is held identical to that baseline -- 15-min EMA(9)/EMA(20)
entry/exit, 1.36xATR(14) stop / 3xATR(14) target, 15:00 IST hard exit,
long+short, same 20-name universe, same Rs 1 Cr / 12.5%-per-trade / max-6-
concurrent sizing, same 2021-07-01 -> 2026-06-30 window -- with ONE change:
the regime filter. Baseline uses EMA(200) on 30-min candles (bullish above,
bearish below). This script replaces it with the JdK RRG-style RS-Ratio /
RS-Momentum regime already used by the separate EMA Regime Crossover Swing
strategy (strategies/backtests/ema_regime_crossover_swing_backtest.py),
computed on DAILY bars against each stock's own index (NIFTY for the Nifty
50 pool, NIFTYNXT50 for the Nifty Next 50 pool).

Baseline being compared against (2026-07-17 run): 10,348 accepted trades,
combined PF 1.06, LONG PF 0.99 (net loser), SHORT PF 1.16 (carries the
whole edge). The question this script answers: does the RRG regime fix
the long-side weakness, and at what cost (if any) to overall PF/trade count.

RS_MOMENTUM_SLOPE_LOOKBACK = 2, not the base swing backtest's 3 -- 2 is the
sweep-validated, spec-locked value from
ema_regime_crossover_swing_long_only_trailing_stop.py, not a leftover
pre-sweep default.

Causality / lookahead: a daily bar only "closes" at 15:30 IST, after that
day's own 15-min bars have already traded. So every 15m bar of day D must
see day D-1's *completed* regime, never day D's still-forming one. The
daily regime series is explicitly shifted by one row (one prior trading
day, via positional .shift(1) -- NOT a calendar shift(freq="D"), since the
daily index skips weekends/holidays) before being forward-filled onto the
15m index. This is the daily analogue of the baseline's own
`regime_series.reindex(df15.index, method="ffill")` causal-ffill of a
same-day-computable 30m regime.

Confound to note before reading results: unlike EMA(200) (strictly binary,
every bar is BULL or BEAR), the RRG regime has a genuine third NEUTRAL
bucket (plus a warmup-unknown period before the RS z-score window fills).
That means the pre-cap signal count will very likely be materially below
baseline's for a structural reason -- fewer bars ever satisfy "regime ==
BULL"/"BEAR" at all -- not purely because of regime quality. The per-symbol
regime-mix print below quantifies this so it isn't mistaken for a
quality signal.
"""

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

ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14
STOP_ATR_MULT = 1.36
TARGET_ATR_MULT = 3.0
HARD_EXIT = dtime(15, 0)
TOTAL_CAPITAL = 10_000_000  # Rs 1 Cr, matching OpenAlgo Sandbox convention
POSITION_PCT = 0.125
MAX_CONCURRENT = 6

# RRG regime params -- identical to ema_regime_crossover_swing_long_only_trailing_stop.py
RS_ZSCORE_WINDOW = 10
RS_SMOOTH_WINDOW = 2
RS_MOMENTUM_SLOPE_LOOKBACK = 2  # locked/sweep-validated value; base swing_backtest.py uses 3

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10
INDEX_FOR = {**{s: "NIFTY" for s in NIFTY50_TOP10}, **{s: "NIFTYNXT50" for s in NEXT50_TOP10}}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_rrg_intraday")
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
    df.index = df.index.normalize()  # daily bars -- date-only for clean stock/index alignment
    return df.sort_index(), None


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_rs_ratio_momentum(stock_close, index_close):
    """JdK RRG-style RS-Ratio / RS-Momentum, aligned on shared trading dates."""
    rs = 100 * (stock_close / index_close)
    rs_ratio_raw = 100 + (rs - rs.rolling(RS_ZSCORE_WINDOW).mean()) / rs.rolling(RS_ZSCORE_WINDOW).std()
    rs_ratio = rs_ratio_raw.rolling(RS_SMOOTH_WINDOW).mean()
    mom_raw = 100 + (rs_ratio - rs_ratio.rolling(RS_ZSCORE_WINDOW).mean()) / rs_ratio.rolling(RS_ZSCORE_WINDOW).std()
    rs_momentum = mom_raw.rolling(RS_SMOOTH_WINDOW).mean()
    return rs_ratio, rs_momentum


# Fetch both indices once (shared across the whole universe)
index_data = {}
for idx_symbol in ("NIFTY", "NIFTYNXT50"):
    df_idx, err = _fetch_daily(idx_symbol, "NSE_INDEX")
    if err:
        raise SystemExit(f"Could not fetch index {idx_symbol}: {err}")
    index_data[idx_symbol] = df_idx
    print(f"Index {idx_symbol}: {len(df_idx):,} daily bars ({df_idx.index[0].date()} -> {df_idx.index[-1].date()})")

print(f"\nUniverse: {len(UNIVERSE)} names")
all_trades = []
skipped = {}
regime_mix_totals = {"BULL": 0, "BEAR": 0, "NEUTRAL": 0, "unknown": 0}

for symbol in UNIVERSE:
    df15, err15 = _fetch(symbol, "15m")
    if err15:
        skipped[symbol] = f"15m: {err15}"
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    dfd, errd = _fetch_daily(symbol, "NSE")
    if errd:
        skipped[symbol] = f"daily: {errd}"
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    idx_symbol = INDEX_FOR[symbol]
    idx_df = index_data[idx_symbol]

    # Align stock and index on shared trading dates only (inner join via dropna)
    combined_close = pd.DataFrame({"stock": dfd["close"], "index": idx_df["close"]}).dropna()
    if len(combined_close) < RS_ZSCORE_WINDOW + RS_SMOOTH_WINDOW:
        skipped[symbol] = f"insufficient overlapping daily history for RS calc: {len(combined_close)} days, need >={RS_ZSCORE_WINDOW + RS_SMOOTH_WINDOW}"
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    rs_ratio, rs_momentum = compute_rs_ratio_momentum(combined_close["stock"], combined_close["index"])
    dfd["rs_ratio"] = rs_ratio.reindex(dfd.index)
    dfd["rs_momentum"] = rs_momentum.reindex(dfd.index)

    rs_momentum_rising = dfd["rs_momentum"] > dfd["rs_momentum"].shift(RS_MOMENTUM_SLOPE_LOOKBACK)
    rs_momentum_falling = dfd["rs_momentum"] < dfd["rs_momentum"].shift(RS_MOMENTUM_SLOPE_LOOKBACK)
    bull_regime = (dfd["rs_ratio"] > 100) & ((dfd["rs_momentum"] > 100) | rs_momentum_rising)
    bear_regime = (dfd["rs_ratio"] < 100) & ((dfd["rs_momentum"] < 100) | rs_momentum_falling)
    dfd["regime"] = np.where(bull_regime, "BULL", np.where(bear_regime, "BEAR", "NEUTRAL"))

    # Rows still in the RS z-score warmup window are NaN in rs_ratio/rs_momentum;
    # np.where() treats NaN comparisons as False, which would silently mislabel
    # them "NEUTRAL". Mask those back to unknown so the sanity print below can
    # tell "not yet warmed up" apart from "genuinely neutral regime".
    warmup = dfd["rs_ratio"].isna() | dfd["rs_momentum"].isna()
    dfd.loc[warmup, "regime"] = np.nan

    # Regime-mix sanity print (pre-shift distribution -- shifting doesn't change
    # the aggregate proportions, only which day each value applies to)
    counts = dfd["regime"].value_counts(dropna=False)
    total_days = len(dfd)
    n_bull = int(counts.get("BULL", 0))
    n_bear = int(counts.get("BEAR", 0))
    n_neutral = int(counts.get("NEUTRAL", 0))
    n_unknown = int(counts.get(np.nan, 0)) if counts.index.hasnans else total_days - n_bull - n_bear - n_neutral
    print(f"  {symbol} (vs {idx_symbol}) regime mix: "
          f"BULL {n_bull/total_days*100:.1f}% BEAR {n_bear/total_days*100:.1f}% "
          f"NEUTRAL {n_neutral/total_days*100:.1f}% unknown/warmup {n_unknown/total_days*100:.1f}%")
    regime_mix_totals["BULL"] += n_bull
    regime_mix_totals["BEAR"] += n_bear
    regime_mix_totals["NEUTRAL"] += n_neutral
    regime_mix_totals["unknown"] += n_unknown

    # Lookahead-avoidance: shift the daily regime by one row (one prior trading
    # day) before broadcasting onto 15m bars -- every 15m bar of day D sees
    # day D-1's *completed* regime, never day D's still-forming one.
    dfd["regime_effective"] = dfd["regime"].shift(1)

    df15["ema_fast"] = df15["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df15["ema_slow"] = df15["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df15["atr"] = compute_atr(df15)
    df15["bull_cross"] = (df15["ema_fast"].shift(1) <= df15["ema_slow"].shift(1)) & (df15["ema_fast"] > df15["ema_slow"])
    df15["bear_cross"] = (df15["ema_fast"].shift(1) >= df15["ema_slow"].shift(1)) & (df15["ema_fast"] < df15["ema_slow"])

    # map each 15m bar to the prior completed daily regime reading (causal, no lookahead)
    df15["regime"] = dfd["regime_effective"].reindex(df15.index, method="ffill")

    print(f"  {symbol}: {len(df15):,} 15m bars, {len(dfd):,} daily bars "
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
            continue  # no same-day next bar (last bar of the day) -- skip
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

total_regime_days = sum(regime_mix_totals.values())
if total_regime_days:
    print(f"\nUniverse-wide regime mix: "
          f"BULL {regime_mix_totals['BULL']/total_regime_days*100:.1f}% "
          f"BEAR {regime_mix_totals['BEAR']/total_regime_days*100:.1f}% "
          f"NEUTRAL {regime_mix_totals['NEUTRAL']/total_regime_days*100:.1f}% "
          f"unknown/warmup {regime_mix_totals['unknown']/total_regime_days*100:.1f}%")

if not all_trades:
    print(f"\nNo trades generated at all. Skipped symbols: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nTotal signals across universe (before position-cap filtering): {len(trades_df)}")

# ---- Portfolio-level simulation: fixed 12.5% sizing, max 6 concurrent positions ----
position_size = TOTAL_CAPITAL * POSITION_PCT
open_positions = []  # list of exit_time, still "open" until that time passes
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
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_rrg_intraday_trades.csv"), index=False)

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
    print(f"\n=== EMA Regime Crossover -- RRG Regime (Intraday) Backtest ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L: {avg_pct:+.3f}% | PF: {pf:.2f}")
    print(f"Total P&L: Rs {total_rupees:+,.0f} on Rs {TOTAL_CAPITAL:,.0f} capital")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")

    print(f"\nBy direction (vs. EMA200 baseline: combined PF 1.06, LONG PF 0.99, SHORT PF 1.16):")
    for direction in ("LONG", "SHORT"):
        sub = result_df[result_df["direction"] == direction]
        if sub.empty:
            print(f"  {direction}: no trades")
            continue
        n_d = len(sub)
        wr_d = (sub["pnl_pct"] > 0).sum() / n_d * 100
        gw_d = sub[sub["pnl_pct"] > 0]["pnl_rupees"].sum()
        gl_d = abs(sub[sub["pnl_pct"] <= 0]["pnl_rupees"].sum())
        pf_d = gw_d / gl_d if gl_d > 0 else float("inf")
        total_d = sub["pnl_rupees"].sum()
        print(f"  {direction}: {n_d} trades | WR {wr_d:.1f}% | PF {pf_d:.2f} | Total P&L Rs {total_d:+,.0f}")

    print(f"\nPer-symbol trade counts:")
    print(result_df["symbol"].value_counts().to_string())
