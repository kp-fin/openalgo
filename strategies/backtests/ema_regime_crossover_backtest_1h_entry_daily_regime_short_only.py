"""
EMA Regime Crossover -- 1-Hour Entry Timeframe + Daily Regime, SHORT-only
(2026-08-03, Karan's request: "what if intraday timeframe goes from 15m to 1h?")

Base config is test #8 exactly (ema_regime_crossover_backtest_narrow_window_wide_stop_short_only.py):
narrow window 11:45-12:45, SHORT-only, 2.2xATR stop / 3.0xATR target, REVERSE_CROSS,
15:00 hard exit, 59-name universe (full Nifty 50 + top-10 Nifty Next 50), same
charges model, same capital-slice-then-leverage sizing, same daily circuit breakers.

Two changes, per Karan's explicit confirmation this session:
  1. Entry/exit timeframe: 15-min -> 1-hour. EMA(9)/EMA(20) crossover, ATR(14),
     stop/target multiples all recomputed on 1h bars instead of 15m.
  2. Regime timeframe scaled proportionally: 30-min EMA(200) -> daily EMA(200),
     keeping the original ~2:1 regime-slower-than-entry ratio. The daily EMA is
     shifted by 1 trading day before being forward-filled onto the 1h index --
     the same causal precaution used in the 2026-07-20 RRG regime A/B test --
     so no bar ever sees a same-day-incomplete daily candle.

Known risk flagged before running (Karan's answer: widen the window if trades
collapse): the 11:45-12:45 clock window was tuned for 15-min bar granularity.
On 1h bars it only lines up with an entry stemming from the 11:15 IST bar's
crossover (closing/entering at 12:15) -- every other signal falls outside the
window. This script runs the narrow window FIRST; if total accepted trades
fall below MIN_TRADES_FLOOR, it automatically re-runs with the window widened
to the full session (09:15-15:00) and reports both results side by side.
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
STOP_ATR_MULT = 2.2
TARGET_ATR_MULT = 3.0
HARD_EXIT = dtime(15, 0)

NARROW_WINDOW_START = dtime(11, 45)
NARROW_WINDOW_END = dtime(12, 45)
WIDE_WINDOW_START = dtime(9, 15)
WIDE_WINDOW_END = dtime(15, 0)
MIN_TRADES_FLOOR = 30  # below this, narrow window is treated as collapsed

SHORT_ONLY = True

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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_1h_entry_daily_regime")
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


def generate_signals(window_start, window_end):
    """Fetch 1h + daily bars per symbol, build the signal candidate list for a given entry window."""
    all_trades = []
    skipped = {}
    rejected_window = 0
    rejected_direction = 0

    for symbol in UNIVERSE:
        df1h, err1h = _fetch(symbol, "1h")
        if err1h:
            skipped[symbol] = err1h
            continue
        dfd, errd = _fetch(symbol, "D")
        if errd:
            skipped[symbol] = f"daily fetch failed: {errd}"
            continue

        dfd["ema200"] = dfd["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
        dfd["regime"] = np.where(dfd["close"] > dfd["ema200"], "BULL", "BEAR")
        # Shift by 1 trading day so an hourly bar only ever sees YESTERDAY's
        # completed daily regime -- no same-day lookahead, mirrors the
        # 2026-07-20 RRG regime test's own causal handling.
        regime_series = dfd["regime"].shift(1)

        df1h["ema_fast"] = df1h["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
        df1h["ema_slow"] = df1h["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
        df1h["atr"] = compute_atr(df1h)
        df1h["bull_cross"] = (df1h["ema_fast"].shift(1) <= df1h["ema_slow"].shift(1)) & (df1h["ema_fast"] > df1h["ema_slow"])
        df1h["bear_cross"] = (df1h["ema_fast"].shift(1) >= df1h["ema_slow"].shift(1)) & (df1h["ema_fast"] < df1h["ema_slow"])

        # Map each 1h bar to its own calendar day's regime value (previous day's
        # completed daily bar), via a per-day lookup rather than a naive ffill
        # across the untz'd daily index.
        day_regime = regime_series.copy()
        day_regime.index = day_regime.index.date
        df1h["day"] = df1h.index.date
        df1h["regime"] = df1h["day"].map(day_regime)

        df1h = df1h.dropna(subset=["atr", "regime"])
        for i in range(len(df1h) - 1):
            row = df1h.iloc[i]
            direction = None
            if row["bull_cross"] and row["regime"] == "BULL":
                direction = "LONG"
            elif row["bear_cross"] and row["regime"] == "BEAR":
                direction = "SHORT"
            if direction is None:
                continue
            if SHORT_ONLY and direction == "LONG":
                rejected_direction += 1
                continue

            entry_ts = df1h.index[i + 1]
            if entry_ts.date() != df1h.index[i].date():
                continue
            if not (window_start <= entry_ts.time() <= window_end):
                rejected_window += 1
                continue

            entry_bar = df1h.iloc[i + 1]
            entry_price = entry_bar["open"]
            atr = row["atr"]
            if direction == "LONG":
                stop_px = entry_price - STOP_ATR_MULT * atr
                target_px = entry_price + TARGET_ATR_MULT * atr
            else:
                stop_px = entry_price + STOP_ATR_MULT * atr
                target_px = entry_price - TARGET_ATR_MULT * atr

            exit_px, exit_ts, reason = None, None, None
            rest = df1h[df1h.index > entry_ts]
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

    return all_trades, skipped, rejected_window, rejected_direction


def run_portfolio_sim(all_trades, label, csv_name):
    if not all_trades:
        print(f"\n[{label}] No trades generated.")
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)

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
    result_df.to_csv(os.path.join(OUT_DIR, csv_name), index=False)

    print(f"\n=== {label} ===")
    print(f"Rejected -- concurrency cap: {rejected_cap} | daily halt: {rejected_halt} | per-symbol block: {rejected_symbol_block}")

    if result_df.empty:
        print("No trades survived portfolio simulation.")
        return None

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

    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")

    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        if len(s) < 2 or s.std(ddof=1) == 0:
            return float("nan")
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    sharpe_gross = sharpe(daily_gross)
    sharpe_net = sharpe(daily_net)
    print(f"Sharpe (gross): {sharpe_gross:.2f}")
    print(f"Sharpe (net)  : {sharpe_net:.2f}   (goal: >= 1.50)")

    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"Max drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")
    print(f"Trade log -> {OUT_DIR}/{csv_name}")

    return {
        "label": label, "trades": n, "wr_gross": wr_gross, "wr_net": wr_net,
        "pf_gross": pf_gross, "pf_net": pf_net, "total_net": total_net,
        "total_charges": total_charges, "sharpe_gross": sharpe_gross, "sharpe_net": sharpe_net,
        "max_dd_pct": abs(max_dd) / ALLOCATED_CAPITAL * 100,
    }


print(f"Universe: {len(UNIVERSE)} names | SHORT_ONLY={SHORT_ONLY} | Entry/exit timeframe: 1h | Regime: daily EMA(200), shifted 1 day")
print(f"\n--- Pass 1: narrow window {NARROW_WINDOW_START}-{NARROW_WINDOW_END} IST (test #8's own window) ---")
narrow_trades, skipped, rej_window, rej_dir = generate_signals(NARROW_WINDOW_START, NARROW_WINDOW_END)
print(f"Skipped (no data): {skipped}")
print(f"Signal candidates surviving direction+window filter: {len(narrow_trades)} "
      f"(rejected direction: {rej_dir}, rejected outside window: {rej_window})")

narrow_summary = run_portfolio_sim(
    narrow_trades, "1h entry / daily regime -- NARROW window (11:45-12:45), SHORT-only",
    "ema_regime_crossover_1h_narrow_window_trades.csv",
)

wide_summary = None
narrow_n = narrow_summary["trades"] if narrow_summary else 0
if narrow_n < MIN_TRADES_FLOOR:
    print(f"\nNarrow-window trade count ({narrow_n}) fell below the {MIN_TRADES_FLOOR}-trade floor "
          f"-- widening to the full session ({WIDE_WINDOW_START}-{WIDE_WINDOW_END} IST) as agreed.")
    wide_trades, skipped_w, rej_window_w, rej_dir_w = generate_signals(WIDE_WINDOW_START, WIDE_WINDOW_END)
    print(f"Skipped (no data): {skipped_w}")
    print(f"Signal candidates surviving direction+window filter: {len(wide_trades)} "
          f"(rejected direction: {rej_dir_w}, rejected outside window: {rej_window_w})")
    wide_summary = run_portfolio_sim(
        wide_trades, "1h entry / daily regime -- WIDE window (09:15-15:00), SHORT-only",
        "ema_regime_crossover_1h_wide_window_trades.csv",
    )
else:
    print(f"\nNarrow-window trade count ({narrow_n}) clears the {MIN_TRADES_FLOOR}-trade floor -- no widening needed.")

print("\n=== SUMMARY vs. test #8 baseline (15m entry / 30m regime, same window/direction/universe) ===")
print("Baseline (documented): Trades 8,759 | WR(net) 44.7% | PF(net) 0.96 | Sharpe(net) -0.24 | maxDD(net) 29.8%")
for s in (narrow_summary, wide_summary):
    if s:
        print(f"{s['label']}: Trades {s['trades']} | WR(net) {s['wr_net']:.1f}% | PF(net) {s['pf_net']:.2f} | "
              f"Sharpe(net) {s['sharpe_net']:.2f} | maxDD(net) {s['max_dd_pct']:.1f}%")
