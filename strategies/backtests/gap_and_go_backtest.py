"""
Gap-and-Go with Volume Confirmation -- new strategy concept (2026-08-03)

Karan's brief: "come up with a better intraday strategy" after the entire
17-test + v2 + 1h-timeframe campaign on EMA Regime Crossover consistently found
the SAME failure mode regardless of entry logic (crossover, ORB, ADX-momentum,
wider-R:R hourly) -- fixed per-trade charges dominate a design that scans a
static 59-name universe every day for a mechanical signal. This concept
deliberately breaks that shape: instead of scanning everything for the same
rule daily, it only trades names with a real catalyst (an overnight gap) AND
confirming volume, cutting trade count and raising average conviction per
trade -- the one lever this vault's own evidence (high-beta-universe test,
2026-07-30) has already shown moves net Sharpe in the right direction.

Design (first cut, not yet tuned):
  - Gap filter: |today's open vs prior close| >= GAP_PCT_MIN (1.5%)
  - Volume confirmation: first 15m bar's volume >= VOL_MULT (1.5x) the
    symbol's own trailing 20-session average volume for that same opening
    bar (computed causally -- no lookahead into future sessions)
  - Direction: gap up -> LONG candidate, gap down -> SHORT candidate. Both
    directions tested (unlike EMA Regime Crossover's SHORT-only convention --
    the long-side weakness found there was specific to that entry logic's
    regime filter, not necessarily transferable to a gap-driven setup)
  - Entry: breakout of the opening 15m range in the gap direction, confirmed
    by ENTRY_CUTOFF (10:30 IST); one trade per symbol per day, first
    qualifying breakout only, no trade if the range never breaks by cutoff
  - Stop: opposite side of the opening range; Target: 2x opening-range width
  - Hard exit 15:00 IST
  - Same 59-name universe, same charges model, same capital-slice-then-
    leverage sizing (Rs 2,50,000 / 10% pre-leverage / 5x MIS leverage),
    same 6-position cap and daily circuit breakers as EMA Regime Crossover's
    test #8, for direct apples-to-apples comparison.

Not yet tried in this file's history -- no rejection-memory conflict.
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
FILTER_LONG = os.getenv("GAG_FILTER_LONG", "0") == "1"  # LONG gap-size filter, added 2026-08-06 -- matches currently deployed config when =1
LONG_GAP_MIN = float(os.getenv("GAG_LONG_MIN", "0.03"))  # used when FILTER_LONG=1
LONG_GAP_MAX = float(os.getenv("GAG_LONG_MAX", "0.10"))
# GAG_AB=1: one fetch, then portfolio-sim each LONG band (SHORTs always kept). No live deploy.
RUN_AB = os.getenv("GAG_AB", "0") == "1"
VOL_LOOKBACK_DAYS = int(os.getenv("GAG_VOL_LOOKBACK", "20"))  # trailing avg window for opening-bar volume baseline
IST = pytz.timezone("Asia/Kolkata")

GAP_PCT_MIN = 0.015          # 1.5% minimum overnight gap
VOL_MULT = 1.5               # opening 15m volume must be >= 1.5x trailing avg
ENTRY_CUTOFF = dtime(10, 30)  # opening-range breakout must confirm by this time
TARGET_RANGE_MULT = 2.0       # target = entry + 2x opening-range width
HARD_EXIT = dtime(15, 0)
MARKET_OPEN = dtime(9, 15)
OPENING_BAR_END = dtime(9, 30)  # first 15m bar: 09:15-09:30

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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_and_go")
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


print(f"Universe: {len(UNIVERSE)} names | Gap threshold: {GAP_PCT_MIN*100:.1f}% | "
      f"Volume mult: {VOL_MULT}x | Vol lookback: {VOL_LOOKBACK_DAYS}d | "
      f"LONG filter: {FILTER_LONG} ({LONG_GAP_MIN*100:.1f}-{LONG_GAP_MAX*100:.0f}%) | "
      f"AB={RUN_AB} | Entry cutoff: {ENTRY_CUTOFF} IST")
all_trades = []
skipped = {}
rejected_gap = 0
rejected_volume = 0
rejected_no_breakout = 0

for symbol in UNIVERSE:
    df15, err15 = _fetch(symbol, "15m")
    if err15:
        skipped[symbol] = err15
        print(f"  {symbol:12s}: SKIPPED ({err15})", flush=True)
        continue
    dfd, errd = _fetch(symbol, "D")
    if errd:
        skipped[symbol] = f"daily fetch failed: {errd}"
        print(f"  {symbol:12s}: SKIPPED (daily fetch failed: {errd})", flush=True)
        continue

    dfd = dfd.sort_index()
    dfd["prior_close"] = dfd["close"].shift(1)

    df15["day"] = df15.index.date
    opening_bars = df15[(df15.index.time >= MARKET_OPEN) & (df15.index.time < OPENING_BAR_END)].copy()
    opening_bars = opening_bars.groupby("day").first()  # one row per day: the 09:15-09:30 bar
    opening_bars["vol_avg20"] = opening_bars["volume"].shift(1).rolling(VOL_LOOKBACK_DAYS, min_periods=VOL_LOOKBACK_DAYS).mean()

    dfd_idx = dfd.copy()
    dfd_idx.index = dfd_idx.index.date
    opening_bars = opening_bars.join(dfd_idx[["prior_close"]], how="left")
    opening_bars["gap_pct"] = (opening_bars["open"] - opening_bars["prior_close"]) / opening_bars["prior_close"]

    opening_bars = opening_bars.dropna(subset=["gap_pct", "vol_avg20", "prior_close"])
    qualifying_days = opening_bars[
        (opening_bars["gap_pct"].abs() >= GAP_PCT_MIN) &
        (opening_bars["volume"] >= VOL_MULT * opening_bars["vol_avg20"])
    ]
    n_gap_only = (opening_bars["gap_pct"].abs() >= GAP_PCT_MIN).sum()
    rejected_gap += (len(opening_bars) - n_gap_only)
    rejected_volume += (n_gap_only - len(qualifying_days))

    n_trades_before = len(all_trades)
    for day, qrow in qualifying_days.iterrows():
        direction = "LONG" if qrow["gap_pct"] > 0 else "SHORT"
        # AB mode keeps all LONG gaps (>= GAP_PCT_MIN); band filter applied post-fetch.
        if (not RUN_AB) and FILTER_LONG and direction == "LONG" and not (
            LONG_GAP_MIN <= qrow["gap_pct"] <= LONG_GAP_MAX
        ):
            continue
        or_high, or_low = qrow["high"], qrow["low"]
        or_width = or_high - or_low
        if or_width <= 0:
            continue

        day_bars = df15[(df15["day"] == day) & (df15.index.time >= OPENING_BAR_END)]
        entry_ts, entry_price = None, None
        for ts, bar in day_bars.iterrows():
            if ts.time() > ENTRY_CUTOFF:
                break
            if direction == "LONG" and bar["high"] >= or_high:
                entry_ts, entry_price = ts, max(bar["open"], or_high)
                break
            if direction == "SHORT" and bar["low"] <= or_low:
                entry_ts, entry_price = ts, min(bar["open"], or_low)
                break
        if entry_ts is None:
            rejected_no_breakout += 1
            continue

        if direction == "LONG":
            stop_px = or_low
            target_px = entry_price + TARGET_RANGE_MULT * or_width
        else:
            stop_px = or_high
            target_px = entry_price - TARGET_RANGE_MULT * or_width

        exit_px, exit_ts, reason = None, None, None
        rest = df15[(df15["day"] == day) & (df15.index > entry_ts)]
        for ts2, bar2 in rest.iterrows():
            t = ts2.time()
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
            if t >= HARD_EXIT:
                exit_px, exit_ts, reason = bar2["close"], ts2, "HARD_EXIT"
                break
        if exit_px is None:
            continue

        pnl_pct = (exit_px - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_px) / entry_price
        all_trades.append({
            "symbol": symbol, "direction": direction, "gap_pct": qrow["gap_pct"],
            "entry_time": entry_ts, "entry_price": entry_price, "stop_px": stop_px,
            "target_px": target_px, "exit_time": exit_ts, "exit_price": exit_px,
            "pnl_pct": pnl_pct, "reason": reason,
        })

    n_new = len(all_trades) - n_trades_before
    print(f"  {symbol:12s}: {len(qualifying_days):3d} qualifying gap+volume days -> {n_new:3d} trades taken", flush=True)

if not all_trades:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nSkipped (no data): {skipped}")
print(f"Signal candidates: {len(trades_df)} (rejected on gap size: {rejected_gap}, "
      f"rejected on volume: {rejected_volume}, rejected no breakout by cutoff: {rejected_no_breakout})")

capital_per_trade = ALLOCATED_CAPITAL * POSITION_PCT
buying_power = capital_per_trade * ASSUMED_MIS_LEVERAGE
daily_loss_limit = DAILY_LOSS_PCT * ALLOCATED_CAPITAL


def sharpe(s):
    if len(s) < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return s.mean() / s.std(ddof=1) * np.sqrt(252)


def apply_long_band(df, band):
    """band=None keeps all; else (lo, hi) keeps SHORTs + LONGs in [lo, hi]."""
    if band is None:
        return df
    lo, hi = band
    keep = (df["direction"] == "SHORT") | (
        (df["direction"] == "LONG") & (df["gap_pct"] >= lo) & (df["gap_pct"] <= hi)
    )
    return df.loc[keep].sort_values("entry_time").reset_index(drop=True)


def simulate_portfolio(cands):
    pending_exits = []
    seq = 0
    open_count = 0
    current_day = None
    daily_realized_pnl = 0.0
    daily_loss_halted = False
    daily_blocked = set()
    accepted = []
    rejected_cap = rejected_halt = rejected_symbol_block = 0

    for _, cand in cands.iterrows():
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

    return (
        pd.DataFrame(accepted),
        {"cap": rejected_cap, "halt": rejected_halt, "symbol_block": rejected_symbol_block},
    )


def metrics(result_df):
    if result_df.empty:
        return {
            "n": 0, "long_n": 0, "short_n": 0, "wr_net": float("nan"),
            "pf_net": float("nan"), "net_pnl": 0.0, "sharpe_net": float("nan"),
            "max_dd_pct": float("nan"),
        }
    n = len(result_df)
    wr_net = (result_df["net_pnl_rupees"] > 0).mean() * 100
    nw = result_df[result_df["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
    nl = abs(result_df[result_df["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
    pf_net = nw / nl if nl > 0 else float("inf")
    total_net = result_df["net_pnl_rupees"].sum()
    rdf = result_df.copy()
    rdf["exit_time"] = pd.to_datetime(rdf["exit_time"])
    rdf["exit_date"] = rdf["exit_time"].dt.date
    daily_net = rdf.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    dd = rdf.sort_values("exit_time").reset_index(drop=True)
    dd["cum"] = dd["net_pnl_rupees"].cumsum()
    dd["peak"] = dd["cum"].cummax()
    dd["dd"] = dd["cum"] - dd["peak"]
    max_dd_pct = abs(dd["dd"].min()) / ALLOCATED_CAPITAL * 100
    return {
        "n": n,
        "long_n": int((rdf["direction"] == "LONG").sum()),
        "short_n": int((rdf["direction"] == "SHORT").sum()),
        "wr_net": wr_net,
        "pf_net": pf_net,
        "net_pnl": total_net,
        "sharpe_net": sharpe(daily_net),
        "max_dd_pct": max_dd_pct,
    }


def print_metrics(label, result_df, rej):
    m = metrics(result_df)
    print(f"\n=== {label} ===")
    print(f"Rejected -- concurrency cap: {rej['cap']} | daily halt: {rej['halt']} | "
          f"per-symbol block: {rej['symbol_block']}")
    if result_df.empty:
        print("No trades survived portfolio simulation.")
        return m
    print(f"Trades: {m['n']} (LONG {m['long_n']} / SHORT {m['short_n']})")
    print(f"WR (net): {m['wr_net']:.1f}% | PF (net): {m['pf_net']:.2f} | "
          f"Net P&L: Rs {m['net_pnl']:+,.0f}")
    print(f"Sharpe (net): {m['sharpe_net']:.2f} | Max DD (net): {m['max_dd_pct']:.1f}%")
    print("By direction (net):")
    for direction, g in result_df.groupby("direction"):
        gw2 = g[g.net_pnl_rupees > 0]["net_pnl_rupees"].sum()
        gl2 = abs(g[g.net_pnl_rupees <= 0]["net_pnl_rupees"].sum())
        pf2 = gw2 / gl2 if gl2 > 0 else float("inf")
        print(f"  {direction:6s} n={len(g):5d}  WR(net)={(g.net_pnl_rupees>0).mean()*100:5.1f}%  "
              f"PF(net)={pf2:.2f}  total_net=Rs{g.net_pnl_rupees.sum():+10,.0f}")
    return m


# A/B: one fetch, re-run portfolio sim per LONG band (SHORTs unchanged).
AB_BANDS = [
    ("unfiltered (LONG gap>=1.5%, no upper cap)", None),
    ("LONG 1.5-10%", (0.015, 0.10)),
    ("LONG 2-10%", (0.02, 0.10)),
    ("LONG 3-10% (deployed)", (0.03, 0.10)),
]

if RUN_AB:
    candidates_csv = os.path.join(OUT_DIR, f"gap_and_go_candidates_vol{VOL_LOOKBACK_DAYS}.csv")
    trades_df.to_csv(candidates_csv, index=False)
    print(f"\nCandidates (pre-portfolio) -> {candidates_csv}")
    rows = []
    for label, band in AB_BANDS:
        subset = apply_long_band(trades_df, band)
        result_df, rej = simulate_portfolio(subset)
        m = print_metrics(label, result_df, rej)
        rows.append({"band": label, **m})
        tag = "unfiltered" if band is None else f"long{int(band[0]*1000):03d}_{int(band[1]*100)}"
        out = os.path.join(OUT_DIR, f"gap_and_go_trades_vol{VOL_LOOKBACK_DAYS}_{tag}.csv")
        if not result_df.empty:
            result_df.to_csv(out, index=False)
            if band is None and VOL_LOOKBACK_DAYS == 20:
                result_df.to_csv(os.path.join(OUT_DIR, "gap_and_go_trades.csv"), index=False)
            if band == (0.03, 0.10) and VOL_LOOKBACK_DAYS == 20:
                result_df.to_csv(os.path.join(OUT_DIR, "gap_and_go_trades_vol20_filtered.csv"), index=False)

    ab_df = pd.DataFrame(rows)
    ab_csv = os.path.join(OUT_DIR, f"gap_and_go_long_band_ab_vol{VOL_LOOKBACK_DAYS}.csv")
    ab_df.to_csv(ab_csv, index=False)
    print("\n=== A/B SUMMARY (LONG band only; SHORTs identical across rows) ===")
    print(ab_df.to_string(index=False))
    print(f"\nA/B summary -> {ab_csv}")
else:
    result_df, rej = simulate_portfolio(trades_df)
    _tag = f"vol{VOL_LOOKBACK_DAYS}" + ("_filtered" if FILTER_LONG else "")
    trades_csv = os.path.join(OUT_DIR, f"gap_and_go_trades_{_tag}.csv")
    result_df.to_csv(trades_csv, index=False)
    if VOL_LOOKBACK_DAYS == 20 and not FILTER_LONG:
        result_df.to_csv(os.path.join(OUT_DIR, "gap_and_go_trades.csv"), index=False)
    label = (
        f"Gap-and-Go LONG {LONG_GAP_MIN*100:.1f}-{LONG_GAP_MAX*100:.0f}%"
        if FILTER_LONG else "Gap-and-Go unfiltered"
    )
    print_metrics(label, result_df, rej)
    if not result_df.empty:
        print(f"\nTrade log -> {trades_csv}")
