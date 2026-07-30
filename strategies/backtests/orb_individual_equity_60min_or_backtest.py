"""
Opening-Range Breakout -- Intraday Equity, Individual Stocks

Second new strategy concept (Karan, 2026-07-30), genuinely different in
kind from EMA Regime Crossover: classic opening-range breakout (ORB) on
individual NSE cash-equity names, rather than the index-level ORB already
in `indices-system/`. The opening range is the first 30 minutes of the
session (09:15-09:45 IST, i.e. the first two 15m bars). If price later
breaks above the OR high, go LONG (breakout continuation); if it breaks
below the OR low, go SHORT. This is a breakout/momentum signal (not mean
reversion like the VWAP variant tested alongside this one, and not an EMA
crossover regime filter like the existing live strategy).

Entry window: any breakout from 09:45 (OR complete) to 14:30 IST (leaves
room to develop before the 15:00 hard exit) triggers the FIRST breakout of
the day only per symbol (no re-entry after a stop-out, to avoid
chasing chop around the OR boundary repeatedly).

Exit: fixed ATR-scaled stop (opposite side of OR, or ATR-based, whichever
is tighter -- protects against a small OR range giving an unrealistically
tight stop on a volatile name) and ATR-scaled target, hard exit 15:00 IST.

Universe, sizing, portfolio-level concurrency/circuit-breaker simulation,
and charges model are all copied verbatim from the EMA Regime Crossover
series this session for direct comparability.
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

ATR_PERIOD = 14
OR_END = dtime(10, 15)         # opening range = 09:15-10:15 (first four 15m bars) -- widened from 30min
ENTRY_WINDOW_START = dtime(10, 15)
ENTRY_WINDOW_END = dtime(14, 30)
STOP_ATR_MULT = 1.0
TARGET_ATR_MULT = 2.0
HARD_EXIT = dtime(15, 0)

ALLOCATED_CAPITAL = 250_000
POSITION_PCT = 0.10
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT = 6
DAILY_LOSS_PCT = 0.02

NIFTY_50_FULL = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BAJAJFINSV", "AXISBANK", "BHARTIARTL",
    "BAJAJ-AUTO", "BAJFINANCE", "BEL", "ASIANPAINT", "DRREDDY", "EICHERMOT",
    "COALINDIA", "CIPLA", "HDFCBANK", "GRASIM", "HCLTECH", "HDFCLIFE", "HINDUNILVR",
    "ICICIBANK", "INFY", "HINDALCO", "INDIGO", "ITC", "JSWSTEEL", "M&M", "KOTAKBANK",
    "LT", "MARUTI", "MAXHEALTH", "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "SBILIFE",
    "SHRIRAMFIN", "SBIN", "RELIANCE", "TATASTEEL", "SUNPHARMA", "TCS", "TECHM",
    "TATACONSUM", "TMPV", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ETERNAL", "JIOFIN",
]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY_50_FULL + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_individual_equity_60min_or")
os.makedirs(OUT_DIR, exist_ok=True)

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


print(f"Universe: {len(UNIVERSE)} names | OR: 09:15-{OR_END} | Entry window: {ENTRY_WINDOW_START} - {ENTRY_WINDOW_END} IST")
all_trades = []
skipped = {}

for symbol in UNIVERSE:
    df15, err15 = _fetch(symbol, "15m")
    if err15:
        skipped[symbol] = err15
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    df15["atr"] = compute_atr(df15)
    df15 = df15.dropna(subset=["atr"])
    print(f"  {symbol}: {len(df15):,} 15m bars ({df15.index[0].date()} -> {df15.index[-1].date()})")

    df15["date"] = df15.index.date
    for day, day_df in df15.groupby("date"):
        or_bars = day_df[day_df.index.time < OR_END]
        if or_bars.empty:
            continue
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()

        rest_of_day = day_df[day_df.index.time >= ENTRY_WINDOW_START]
        if rest_of_day.empty:
            continue

        traded_today = False
        for i in range(len(rest_of_day)):
            if traded_today:
                break
            bar = rest_of_day.iloc[i]
            t = bar.name.time()
            if t > ENTRY_WINDOW_END:
                break

            direction = None
            if bar["high"] > or_high:
                direction = "LONG"
                breakout_px = or_high
            elif bar["low"] < or_low:
                direction = "SHORT"
                breakout_px = or_low
            if direction is None:
                continue

            entry_price = max(bar["open"], breakout_px) if direction == "LONG" else min(bar["open"], breakout_px)
            atr = bar["atr"]
            if atr <= 0 or np.isnan(atr):
                continue
            or_range = or_high - or_low
            stop_dist = min(STOP_ATR_MULT * atr, or_range) if or_range > 0 else STOP_ATR_MULT * atr
            if direction == "LONG":
                stop_px = entry_price - stop_dist
                target_px = entry_price + TARGET_ATR_MULT * atr
            else:
                stop_px = entry_price + stop_dist
                target_px = entry_price - TARGET_ATR_MULT * atr

            entry_ts = bar.name
            exit_px, exit_ts, reason = None, None, None
            rest = rest_of_day[rest_of_day.index >= entry_ts].iloc[1:]
            for ts2, bar2 in rest.iterrows():
                t2 = ts2.time()
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
                if t2 >= HARD_EXIT:
                    exit_px, exit_ts, reason = bar2["close"], ts2, "HARD_EXIT"
                    break
            if exit_px is None:
                # breakout bar itself was the last bar of the day/window -- close at bar's own close
                exit_px, exit_ts, reason = bar["close"], entry_ts, "HARD_EXIT"

            pnl_pct = (exit_px - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_px) / entry_price
            all_trades.append({
                "symbol": symbol, "direction": direction, "entry_time": entry_ts,
                "entry_price": entry_price, "stop_px": stop_px, "target_px": target_px,
                "exit_time": exit_ts, "exit_price": exit_px, "pnl_pct": pnl_pct, "reason": reason,
            })
            traded_today = True

if not all_trades:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nSignal candidates: {len(trades_df)}")

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
result_df.to_csv(os.path.join(OUT_DIR, "orb_individual_equity_60min_or_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | daily halt: {rejected_halt} | per-symbol block: {rejected_symbol_block}")

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

    print(f"\n=== Opening-Range Breakout -- Individual Equity ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")

    print("\nBy exit reason (net of charges):")
    for reason, g in result_df.groupby("reason"):
        print(f"  {reason:14s} n={len(g):5d} ({len(g)/n*100:4.1f}%)  WR={(g.net_pnl_rupees>0).mean()*100:5.1f}%  "
              f"avg_pnl_pct={g.pnl_pct.mean()*100:+.3f}%  total_net=Rs{g.net_pnl_rupees.sum():+10,.0f}")

    print("\nBy direction (net of charges):")
    for d, g in result_df.groupby("direction"):
        w = g.loc[g.net_pnl_rupees > 0, "net_pnl_rupees"].sum()
        l = -g.loc[g.net_pnl_rupees < 0, "net_pnl_rupees"].sum()
        pf_d = w / l if l > 0 else float("inf")
        print(f"  {d}: n={len(g)}, WR={((g.net_pnl_rupees>0).mean()*100):.1f}%, "
              f"net_pnl={g.net_pnl_rupees.sum():+,.0f}, PF(net)={pf_d:.2f}")

    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    print(f"\nSharpe (gross): {sharpe(daily_gross):.2f}")
    print(f"Sharpe (net)  : {sharpe(daily_net):.2f}   (goal: >= 1.50)")

    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"\nMax drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")

    print(f"\nTrade log -> {OUT_DIR}/orb_individual_equity_60min_or_trades.csv")
