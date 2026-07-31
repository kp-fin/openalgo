"""
EMA Regime Crossover v2 -- Concept C: ADX-Confirmed Momentum Continuation, One Trade/Day/Symbol

Concepts A (hourly crossover, daily regime, wide 1:3 R:R) and B (opening-range
breakout, daily regime, wide 1:3 R:R) both came back net-negative (Sharpe net
-2.78 and -2.17) -- and both showed the SAME failure mode: HARD_EXIT dominated
exits (95% of A's trades, 80% of B's), meaning the wide stop/target distance
almost never actually triggered intraday. The strategy was effectively betting
on the day's open-to-close price change, a near-50/50 coin flip, while still
paying full round-trip charges on every trade. Widening R:R without checking
whether the stop/target can realistically complete intraday just re-introduces
a different version of the same cost problem.

Concept C's fix: keep the stop/target CLOSE ENOUGH to the entry (relative to
the instrument's own recent volatility) that they actually resolve most days,
while cutting FREQUENCY through a genuinely different entry basis -- momentum
CONTINUATION confirmed by ADX, not a crossover or a breakout level, and a hard
cap of ONE trade per symbol per day (vs. the original design's unlimited
same-direction re-entries, and Concepts A/B's implicit multiple-signals-per-
day exposure).

Design:
  - Regime: daily EMA(50), shifted 1 day (same regime source as Concepts A/B,
    for a clean three-way comparison).
  - Trend-strength filter: 15-min ADX(14) > 25 (standard "trending" threshold)
    AND +DI > -DI (for LONG) / -DI > +DI (for SHORT) -- confirms an established,
    strengthening move already underway, rather than trying to catch the start
    of one.
  - Entry: the FIRST 15-min bar in the 09:30-13:00 window where ADX>25, the DI
    alignment matches direction, AND direction matches the day's regime. Enter
    at that bar's close. At most one trade per symbol per day (first
    qualifying signal only -- no re-entry same day, in either direction).
  - Stop: 1.5xATR(14) on 15m (tighter than Concepts A/B, close enough to
    realistically trigger on a failed continuation).
  - Target: 3xATR(14) on 15m (1:2 R:R -- less extreme than A/B's 1:3, chosen
    deliberately so the target is reachable within a single session rather
    than requiring an outsized intraday move).
  - No early exit besides stop/target/hard-exit -- consistent with Concepts
    A/B and the 17-test campaign's own trailing-target finding.
  - Hard exit 15:00 IST, unchanged.

Same universe, same capital-slice-then-leverage sizing, same 6-position
concurrency cap, same charges model applied immediately -- identical
methodology to Concepts A and B for a clean comparison.
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

REGIME_EMA = 50
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 3.0     # 1:2 R:R -- deliberately less extreme than Concepts A/B
ENTRY_WINDOW_START = dtime(9, 30)
ENTRY_WINDOW_END = dtime(13, 0)
HARD_EXIT = dtime(15, 0)

ALLOCATED_CAPITAL = 250_000
POSITION_PCT = 0.10
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT = 6

BROKERAGE_FLAT = 20.0
BROKERAGE_PCT = 0.0003
STT_PCT = 0.00025
EXCHANGE_PCT = 0.0000297
SEBI_PCT = 0.000001
STAMP_DUTY_PCT = 0.00003
GST_PCT = 0.18


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
    return brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst


NIFTY50 = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
           "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL", "BHARTIARTL",
           "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM",
           "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
           "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL",
           "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
           "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
           "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
           "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = sorted(set(NIFTY50 + NEXT50_TOP10))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_v2_adx_momentum")
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


def compute_adx(df, period=ADX_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


print(f"Universe: {len(UNIVERSE)} names (Concept C: ADX-confirmed momentum continuation, 1 trade/day/symbol)")
all_trades = []
skipped = {}

for symbol in UNIVERSE:
    df15, err15 = _fetch(symbol, "15m")
    if err15:
        skipped[symbol] = err15
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue
    dfd, errd = _fetch(symbol, "D")
    if errd:
        skipped[symbol] = errd
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    dfd["ema"] = dfd["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    dfd["regime"] = np.where(dfd["close"] > dfd["ema"], "BULL", "BEAR")
    dfd["regime_shifted"] = dfd["regime"].shift(1)
    daily_regime = dfd["regime_shifted"].copy()
    daily_regime.index = dfd.index.normalize()

    df15["atr"] = compute_atr(df15)
    df15["adx"], df15["plus_di"], df15["minus_di"] = compute_adx(df15)
    df15["day"] = df15.index.normalize()
    df15["t"] = df15.index.time
    df15["regime"] = df15["day"].map(daily_regime)

    print(f"  {symbol}: {len(df15):,} 15m bars, {len(dfd):,} daily bars "
          f"({df15.index[0].date()} -> {df15.index[-1].date()})")

    df15 = df15.dropna(subset=["atr", "adx", "regime"])
    for day, day_df in df15.groupby("day"):
        regime = day_df["regime"].iloc[0]
        if regime is None or pd.isna(regime):
            continue
        window = day_df[(day_df["t"] >= ENTRY_WINDOW_START) & (day_df["t"] <= ENTRY_WINDOW_END)]
        direction = None
        entry_ts = entry_price = atr = None
        for ts, bar in window.iterrows():
            if bar["adx"] <= ADX_THRESHOLD:
                continue
            if regime == "BULL" and bar["plus_di"] > bar["minus_di"]:
                direction, entry_ts, entry_price, atr = "LONG", ts, bar["close"], bar["atr"]
                break
            if regime == "BEAR" and bar["minus_di"] > bar["plus_di"]:
                direction, entry_ts, entry_price, atr = "SHORT", ts, bar["close"], bar["atr"]
                break
        if direction is None:
            continue

        if direction == "LONG":
            stop_px = entry_price - STOP_ATR_MULT * atr
            target_px = entry_price + TARGET_ATR_MULT * atr
        else:
            stop_px = entry_price + STOP_ATR_MULT * atr
            target_px = entry_price - TARGET_ATR_MULT * atr

        after_entry = day_df[day_df.index > entry_ts]
        exit_px, exit_ts, reason = None, None, None
        for ts2, bar2 in after_entry.iterrows():
            tt = ts2.time()
            if direction == "LONG":
                if bar2["low"] <= stop_px:
                    exit_px, exit_ts, reason = stop_px, ts2, "STOP"
                    break
                if bar2["high"] >= target_px:
                    exit_px, exit_ts, reason = target_px, ts2, "TARGET"
                    break
            else:
                if bar2["high"] >= stop_px:
                    exit_px, exit_ts, reason = stop_px, ts2, "STOP"
                    break
                if bar2["low"] <= target_px:
                    exit_px, exit_ts, reason = target_px, ts2, "TARGET"
                    break
            if tt >= HARD_EXIT:
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
    if qty <= 0:
        continue
    pnl_rupees = qty * trade["entry_price"] * (trade["pnl_pct"] / 100)
    t = trade.to_dict()
    t["qty"] = qty
    t["pnl_rupees"] = round(pnl_rupees, 2)
    t["charges"] = round(compute_charges(trade["entry_price"], trade["exit_price"], qty), 2)
    t["net_pnl_rupees"] = round(t["pnl_rupees"] - t["charges"], 2)
    accepted.append(t)

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_v2_adx_momentum_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected for exceeding {MAX_CONCURRENT}-position cap: {rejected_capacity}")

if result_df.empty:
    print("\nNo trades survived position-cap filtering.")
    raise SystemExit(0)

n = len(result_df)
wr_gross = (result_df["pnl_rupees"] > 0).mean() * 100
wr_net = (result_df["net_pnl_rupees"] > 0).mean() * 100
total_gross = result_df["pnl_rupees"].sum()
total_charges = result_df["charges"].sum()
total_net = result_df["net_pnl_rupees"].sum()

gw = result_df[result_df["pnl_rupees"] > 0]["pnl_rupees"].sum()
gl = abs(result_df[result_df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
pf_gross = gw / gl if gl > 0 else float("inf")
nw = result_df[result_df["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
nl = abs(result_df[result_df["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
pf_net = nw / nl if nl > 0 else float("inf")

print(f"\n=== Concept C: ADX-Confirmed Momentum Continuation / Daily EMA50 Regime / 1:2 R:R / 1 trade/day/symbol ===")
print(f"Trades: {n} | WR gross: {wr_gross:.1f}% | WR net: {wr_net:.1f}%")
print(f"Gross P&L: Rs {total_gross:+,.0f} | PF gross: {pf_gross:.2f}")
print(f"Charges  : Rs {total_charges:,.0f} ({total_charges/n:.0f}/trade avg)")
print(f"Net P&L  : Rs {total_net:+,.0f} | PF net: {pf_net:.2f}")
print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")


def _pf(g, col):
    w = g.loc[g[col] > 0, col].sum()
    losses = -g.loc[g[col] < 0, col].sum()
    return w / losses if losses > 0 else float("inf")


print("\nBy direction:")
for d, g in result_df.groupby("direction"):
    print(f"  {d}: n={len(g)}, WR(net)={((g.net_pnl_rupees>0).mean()*100):.1f}%, "
          f"net_pnl={g.net_pnl_rupees.sum():+,.0f}, PF(net)={_pf(g,'net_pnl_rupees'):.2f}, PF(gross)={_pf(g,'pnl_rupees'):.2f}")

result_df["exit_date"] = pd.to_datetime(result_df["exit_time"]).dt.date
daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL
daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
sharpe_gross = daily_gross.mean() / daily_gross.std() * np.sqrt(252) if daily_gross.std() > 0 else float("nan")
sharpe_net = daily_net.mean() / daily_net.std() * np.sqrt(252) if daily_net.std() > 0 else float("nan")
print(f"\nAnnualised Sharpe (gross): {sharpe_gross:.2f}")
print(f"Annualised Sharpe (net)  : {sharpe_net:.2f}")

dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
dd_df["cum_net"] = dd_df["net_pnl_rupees"].cumsum()
dd_df["peak"] = dd_df["cum_net"].cummax()
dd_df["dd"] = dd_df["cum_net"] - dd_df["peak"]
max_dd = dd_df["dd"].min()
print(f"Max drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")
