"""
EMA Regime Crossover -- Variation: 4 Concurrent Positions / Rs 35,000 per Trade

Explores a sizing variation Karan asked to test: reduce MAX_CONCURRENT from 6 to 4
and raise capital_per_trade from the live script's current Rs 25,000 (10% of
Rs 2,50,000 allocated_capital) to Rs 35,000 flat.

Identical to ema_regime_crossover_backtest_resized.py in every respect except:
  - MAX_CONCURRENT: 6 -> 4
  - capital_per_trade: Rs 25,000 -> Rs 35,000 (implemented as a flat override,
    not tied to allocated_capital %, to isolate the sizing effect cleanly)
  - buying_power = capital_per_trade x ASSUMED_MIS_LEVERAGE (5x), same formula as live

Regime (EMA200/30m), entry (EMA9/20 crossover on 15m), exit (1.36xATR stop /
3xATR target / reverse-cross / 15:00 hard exit), universe, and window are
unchanged from the baseline.

Charges modelled on NSE equity intraday (MIS) cash-segment rates, matching the
manual reconciliation done for the paper-trading P&L on 2026-07-29:
  - Brokerage: Rs 20 flat or 0.03% of turnover, whichever is lower, per executed order
  - STT: 0.025% on sell-side turnover only (intraday equity)
  - Exchange transaction charges (NSE): 0.00297% of turnover (buy + sell)
  - SEBI turnover fee: 0.0001% of turnover (buy + sell)
  - Stamp duty: 0.003% on buy-side turnover only
  - GST: 18% on (brokerage + exchange charges + SEBI fee)
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

# ---- Sizing variation under test ----
CAPITAL_PER_TRADE    = 35_000     # flat override (was Rs 25,000 = 10% of Rs 2,50,000 live)
ASSUMED_MIS_LEVERAGE = 5
MAX_CONCURRENT       = 4           # was 6

# ---- Charges model (NSE equity intraday / MIS) ----
BROKERAGE_FLAT   = 20.0
BROKERAGE_PCT    = 0.0003     # 0.03%
STT_PCT          = 0.00025   # 0.025%, sell-side only
EXCHANGE_PCT     = 0.0000297 # 0.00297%, both sides
SEBI_PCT         = 0.000001  # 0.0001%, both sides
STAMP_DUTY_PCT   = 0.00003   # 0.003%, buy-side only
GST_PCT          = 0.18

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_4pos_35k")
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


def compute_charges(entry_price, exit_price, qty):
    """Per-trade round-trip charges (buy + sell) in Rupees."""
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover

    brokerage = 2 * min(BROKERAGE_FLAT, BROKERAGE_PCT * buy_turnover)  # one leg each side
    stt = STT_PCT * sell_turnover
    exchange_chg = EXCHANGE_PCT * total_turnover
    sebi_chg = SEBI_PCT * total_turnover
    stamp_duty = STAMP_DUTY_PCT * buy_turnover
    gst = GST_PCT * (brokerage + exchange_chg + sebi_chg)

    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_chg": round(exchange_chg, 2),
        "sebi_chg": round(sebi_chg, 2),
        "stamp_duty": round(stamp_duty, 2),
        "gst": round(gst, 2),
        "total_charges": round(total, 2),
    }


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

# ---- Portfolio-level simulation: 4 concurrent positions, Rs 30,000/trade ----
buying_power = CAPITAL_PER_TRADE * ASSUMED_MIS_LEVERAGE
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

    charges = compute_charges(trade["entry_price"], trade["exit_price"], qty)
    net_pnl_rupees = pnl_rupees - charges["total_charges"]

    t = trade.to_dict()
    t["qty"] = qty
    t["gross_pnl_rupees"] = round(pnl_rupees, 2)
    t.update(charges)
    t["net_pnl_rupees"] = round(net_pnl_rupees, 2)
    accepted.append(t)

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_4pos_35k_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected for exceeding {MAX_CONCURRENT}-position cap: {rejected_capacity}")

if result_df.empty:
    print("\nNo trades survived position-cap filtering.")
else:
    n = len(result_df)
    wins = (result_df["pnl_pct"] > 0).sum()
    wr = wins / n * 100
    avg_pct = result_df["pnl_pct"].mean()

    total_gross = result_df["gross_pnl_rupees"].sum()
    total_charges = result_df["total_charges"].sum()
    total_net = result_df["net_pnl_rupees"].sum()

    gw = result_df[result_df["gross_pnl_rupees"] > 0]["gross_pnl_rupees"].sum()
    gl = abs(result_df[result_df["gross_pnl_rupees"] <= 0]["gross_pnl_rupees"].sum())
    pf_gross = gw / gl if gl > 0 else float("inf")

    nw = result_df[result_df["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
    nl = abs(result_df[result_df["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
    pf_net = nw / nl if nl > 0 else float("inf")

    print(f"\n=== EMA Regime Crossover -- 4 positions / Rs 35,000 per trade ===")
    print(f"Trades: {n} | WR: {wr:.1f}% | Avg P&L/trade: {avg_pct:+.3f}%")
    print(f"Capital per trade: Rs {CAPITAL_PER_TRADE:,.0f} | Buying power/trade: Rs {buying_power:,.0f} "
          f"({ASSUMED_MIS_LEVERAGE}x) | Max concurrent: {MAX_CONCURRENT}")
    print(f"\nGross P&L : Rs {total_gross:+,.0f}  | PF (gross): {pf_gross:.2f}")
    print(f"Charges   : Rs {total_charges:,.0f}")
    print(f"Net P&L   : Rs {total_net:+,.0f}  | PF (net):   {pf_net:.2f}")
    print(f"\nCharges breakdown:")
    print(f"  Brokerage     : Rs {result_df['brokerage'].sum():,.0f}")
    print(f"  STT           : Rs {result_df['stt'].sum():,.0f}")
    print(f"  Exchange chg  : Rs {result_df['exchange_chg'].sum():,.0f}")
    print(f"  SEBI chg      : Rs {result_df['sebi_chg'].sum():,.0f}")
    print(f"  Stamp duty    : Rs {result_df['stamp_duty'].sum():,.0f}")
    print(f"  GST           : Rs {result_df['gst'].sum():,.0f}")

    print(f"\nExit breakdown: {result_df['reason'].value_counts().to_dict()}")

    def _pf(g, col):
        w = g.loc[g[col] > 0, col].sum()
        losses = -g.loc[g[col] < 0, col].sum()
        return w / losses if losses > 0 else float("inf")

    print("\nBy direction (net of charges):")
    for d, g in result_df.groupby("direction"):
        print(f"  {d}: n={len(g)}, WR={((g.pnl_pct>0).mean()*100):.1f}%, "
              f"net_pnl={g.net_pnl_rupees.sum():+,.0f}, PF={_pf(g, 'net_pnl_rupees'):.2f}")

    print(f"\nPer-symbol trade counts:")
    print(result_df["symbol"].value_counts().to_string())

    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_4pos_35k_trades.csv")
