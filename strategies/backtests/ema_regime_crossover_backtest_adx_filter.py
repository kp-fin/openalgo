"""
EMA Regime Crossover -- ADX(14) Trend-Strength Filter Variant

Goal (Karan, 2026-07-29): improve win rate towards 50% and Sharpe ratio
>= 1.5 for the live config. Baseline (current live config: 6 concurrent
positions, Rs 25,000/trade = 10% of Rs 2,50,000, 5x leverage, both circuit-
breaker mechanics, NSE equity intraday charges applied):
  WR 38.1% (net) / 40.9% (gross), PF 0.89 (net) / 1.13 (gross),
  Sharpe -1.75 (net) / +1.79 (gross). Net-of-charges Sharpe is nowhere near
  the +1.5 goal -- the entry signal's average edge (~0.02-0.03% gross per
  trade) is too thin relative to per-trade transaction costs (~0.05-0.09%
  round-trip in percentage terms). Closing this gap needs either a
  materially stronger average edge per trade or far fewer, higher-quality
  trades (or both) -- not just a sizing tweak.

Hypothesis under test: many EMA(9)/EMA(20) crossovers fire during choppy/
range-bound conditions where the "regime" (EMA200 on 30m) is technically
BULL or BEAR but the underlying trend has little actual strength -- these
are the low-quality, whipsaw-prone signals responsible for both the low
win rate and thin average edge. ADX(14) is the DhanHQ/ScanX skill's own
standard trend-strength indicator (labelled "Strong Trend" vs "Weak Trend"
in ScanX's Technicals tab) -- filtering entries to require ADX(14) >=
ADX_THRESHOLD on the 30m regime timeframe (same timeframe the regime check
already uses) at the moment of entry should raise win rate and average
edge by excluding weak-trend whipsaws, at the cost of fewer trades.

ADX(14) computed via standard Wilder smoothing on the 30m bars (same
timeframe as the EMA200 regime filter, for consistency and to avoid 15m
noise). Everything else (entry/exit mechanics, universe, window, sizing,
both circuit-breaker mechanics, charges model) is identical to the current
live config's own backtest (ema_regime_crossover_backtest_daily_circuit_
breaker.py) -- this isolates the ADX filter's effect cleanly.
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

ADX_PERIOD = 14
ADX_THRESHOLD = 25  # ScanX's own "Strong Trend" cutoff convention

ALLOCATED_CAPITAL = 250_000
POSITION_PCT = 0.10
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT = 6
DAILY_LOSS_PCT = 0.02

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_adx_filter")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Charges model (NSE equity intraday / MIS, Dhan) -- same as prior scripts ----
BROKERAGE_FLAT = 20.0
BROKERAGE_PCT = 0.0003
STT_PCT = 0.00025
EXCHANGE_PCT = 0.0000297
SEBI_PCT = 0.000001
STAMP_DUTY_PCT = 0.00003
GST_PCT = 0.18

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


def compute_adx(df, period=ADX_PERIOD):
    """Standard Wilder ADX(14)."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx


def compute_charges(entry_price, exit_price, qty):
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover
    brokerage = 2 * min(BROKERAGE_FLAT, BROKERAGE_PCT * buy_turnover)
    stt = STT_PCT * sell_turnover
    exchange_chg = EXCHANGE_PCT * total_turnover
    sebi_chg = SEBI_PCT * total_turnover
    stamp_duty = STAMP_DUTY_PCT * buy_turnover
    gst = GST_PCT * (brokerage + exchange_chg + sebi_chg)
    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst
    return {"brokerage": round(brokerage, 2), "stt": round(stt, 2), "exchange_chg": round(exchange_chg, 2),
            "sebi_chg": round(sebi_chg, 2), "stamp_duty": round(stamp_duty, 2), "gst": round(gst, 2),
            "total_charges": round(total, 2)}


print(f"Universe: {len(UNIVERSE)} names | ADX filter: ADX(14) on 30m >= {ADX_THRESHOLD}")
all_trades = []
skipped = {}
rejected_adx_total = 0

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
    df30["adx"] = compute_adx(df30)
    regime_series = df30["regime"].copy()
    adx_series = df30["adx"].copy()

    df15["ema_fast"] = df15["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df15["ema_slow"] = df15["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df15["atr"] = compute_atr(df15)
    df15["bull_cross"] = (df15["ema_fast"].shift(1) <= df15["ema_slow"].shift(1)) & (df15["ema_fast"] > df15["ema_slow"])
    df15["bear_cross"] = (df15["ema_fast"].shift(1) >= df15["ema_slow"].shift(1)) & (df15["ema_fast"] < df15["ema_slow"])
    df15["regime"] = regime_series.reindex(df15.index, method="ffill")
    df15["adx"] = adx_series.reindex(df15.index, method="ffill")

    print(f"  {symbol}: {len(df15):,} 15m bars ({df15.index[0].date()} -> {df15.index[-1].date()})")

    df15 = df15.dropna(subset=["atr", "regime", "adx"])
    for i in range(len(df15) - 1):
        row = df15.iloc[i]
        direction = None
        if row["bull_cross"] and row["regime"] == "BULL":
            direction = "LONG"
        elif row["bear_cross"] and row["regime"] == "BEAR":
            direction = "SHORT"
        if direction is None:
            continue
        if row["adx"] < ADX_THRESHOLD:
            rejected_adx_total += 1
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
            "adx_at_signal": round(row["adx"], 1),
        })

if not all_trades:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nSignal candidates surviving ADX>={ADX_THRESHOLD} filter: {len(trades_df)} "
      f"(rejected on ADX: {rejected_adx_total})")

# ---- Event-driven portfolio simulation: concurrency cap + both circuit-breaker mechanics ----
capital_per_trade = ALLOCATED_CAPITAL * POSITION_PCT
buying_power = capital_per_trade * ASSUMED_MIS_LEVERAGE
daily_loss_limit = DAILY_LOSS_PCT * ALLOCATED_CAPITAL

pending_exits = []
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

    while pending_exits and pending_exits[0][0] <= cand["entry_time"]:
        ex_time, _, ex_trade = heapq.heappop(pending_exits)
        if ex_time.date() == current_day:
            daily_realized_pnl += ex_trade["pnl_rupees"]
            if ex_trade["pnl_rupees"] < 0:
                daily_blocked.add(ex_trade["symbol"])
        open_count -= 1

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
    charges = compute_charges(cand["entry_price"], cand["exit_price"], qty)
    t.update(charges)
    t["net_pnl_rupees"] = round(pnl_rupees - charges["total_charges"], 2)
    accepted.append(t)
    open_count += 1
    seq += 1
    heapq.heappush(pending_exits, (cand["exit_time"], seq, t))

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_adx_filter_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | daily halt: {rejected_halt} | "
      f"per-symbol block: {rejected_symbol_block}")

if result_df.empty:
    print("\nNo trades survived.")
else:
    n = len(result_df)
    wr_gross = (result_df["pnl_rupees"] > 0).mean() * 100
    wr_net = (result_df["net_pnl_rupees"] > 0).mean() * 100
    total_gross = result_df["pnl_rupees"].sum()
    total_charges = result_df["total_charges"].sum()
    total_net = result_df["net_pnl_rupees"].sum()

    gw = result_df[result_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
    pf_gross = gw / gl if gl > 0 else float("inf")
    nw = result_df[result_df["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
    nl = abs(result_df[result_df["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
    pf_net = nw / nl if nl > 0 else float("inf")

    print(f"\n=== EMA Regime Crossover -- ADX(14)>={ADX_THRESHOLD} Filter ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")

    def _pf(g, col):
        w = g.loc[g[col] > 0, col].sum()
        losses = -g.loc[g[col] < 0, col].sum()
        return w / losses if losses > 0 else float("inf")

    print("\nBy direction (net of charges):")
    for d, g in result_df.groupby("direction"):
        print(f"  {d}: n={len(g)}, WR={((g.net_pnl_rupees>0).mean()*100):.1f}%, "
              f"net_pnl={g.net_pnl_rupees.sum():+,.0f}, PF(net)={_pf(g,'net_pnl_rupees'):.2f}")

    # Sharpe (net + gross), daily aggregated
    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    print(f"\nSharpe (gross): {sharpe(daily_gross):.2f}")
    print(f"Sharpe (net)  : {sharpe(daily_net):.2f}   (goal: >= 1.50)")

    # Drawdown net of charges
    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"\nMax drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")

    print(f"\nPer-symbol trade counts:")
    print(result_df["symbol"].value_counts().to_string())
    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_adx_filter_trades.csv")
