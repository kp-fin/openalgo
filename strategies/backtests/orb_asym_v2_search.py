"""
ORB_Asym v2 search — Defensive OR-Hold + nearer BWB bodies.

Optimised: one history fetch, precompute per-trade spot path, score config
grid offline. Gates: Sharpe > 2.5, WR >= 45%.
"""

import os
import sys
import warnings
from datetime import time as dtime
from itertools import product

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE, END_DATE = "2021-07-01", "2026-06-30"
OR_MIN, OR_MAX = 30, 80
ENTRY_END = dtime(12, 0)
HARD_EXIT = dtime(15, 15)
HOLD_BARS = 3
IST = pytz.timezone("Asia/Kolkata")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "orb_asym")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE = os.path.join(OUT_DIR, "nifty_5m_cache.pkl")

SHARPE_GATE = 2.5
WR_GATE = 45.0


def butterfly_intrinsic(delta, body, far):
    return (
        max(delta - 0.0, 0.0)
        - 2.0 * max(delta - body, 0.0)
        + max(delta - far, 0.0)
    )


def sharpe_daily(pnl_by_day):
    if len(pnl_by_day) < 2 or pnl_by_day.std(ddof=1) == 0:
        return float("nan")
    rets = pnl_by_day / 50.0
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


def metrics_from_pnl(days, pnls):
    s = pd.Series(pnls)
    n = len(s)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "pf": float("nan"), "avg": float("nan"),
                "total": 0.0, "sharpe": float("nan")}
    gw = s[s > 0].sum()
    gl = abs(s[s <= 0].sum())
    daily = pd.DataFrame({"day": days, "pnl": pnls}).groupby("day")["pnl"].sum()
    return {
        "n": n,
        "wr": float((s > 0).mean() * 100),
        "pf": float(gw / gl) if gl > 0 else float("inf"),
        "avg": float(s.mean()),
        "total": float(s.sum()),
        "sharpe": sharpe_daily(daily),
    }


# ── Load / cache bars ─────────────────────────────────────────────────────────
if os.path.exists(CACHE):
    print(f"Loading cache {CACHE}", flush=True)
    df = pd.read_pickle(CACHE)
else:
    from openalgo import api as openalgo_api
    print(f"Fetching 5m NIFTY {START_DATE}→{END_DATE} ...", flush=True)
    client = openalgo_api(api_key=API_KEY, host=HOST)
    resp = client.history(symbol="NIFTY", exchange="NSE_INDEX", interval="5m",
                          start_date=START_DATE, end_date=END_DATE)
    df = resp if hasattr(resp, "empty") else pd.DataFrame(resp.get("data", []))
    if "datetime" in getattr(df, "columns", []):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize("Asia/Kolkata") if df.index.tz is None else df.index.tz_convert(IST)
    df = df.sort_index()
    df.to_pickle(CACHE)
    print(f"Cached {len(df):,} bars -> {CACHE}", flush=True)

print(f"Bars: {len(df):,}", flush=True)

daily = df.resample("1D").agg({"open": "first", "close": "last"}).dropna()
daily["prev_net"] = daily["close"].pct_change().shift(1)
prev_net_map = {d.date(): v for d, v in daily["prev_net"].items()}


def generate_hold_signals(require_quiet_prev=False, quiet_thr=0.0042,
                          require_breakout_side=False):
    records = []
    for day, grp in df.groupby(df.index.date):
        orw = grp.between_time("09:15", "09:29")
        if len(orw) < 2:
            continue
        oh, ol = float(orw["high"].max()), float(orw["low"].min())
        ow = oh - ol
        if ow < OR_MIN or ow > OR_MAX:
            continue
        at = grp.between_time("10:15", "10:15")
        if not at.empty:
            c = float(at["close"].iloc[0])
            if ol < c < oh:
                continue
        if require_quiet_prev:
            pn = prev_net_map.get(day)
            if pn is None or abs(pn) > quiet_thr:
                continue
        mid = (oh + ol) / 2.0
        sig = grp.between_time("09:45", "12:00")
        if len(sig) < HOLD_BARS:
            continue
        lows, highs, closes, idx = sig["low"].values, sig["high"].values, sig["close"].values, sig.index
        bull_done = bear_done = False
        for i in range(HOLD_BARS - 1, len(sig)):
            if idx[i].time() > ENTRY_END:
                break
            if (not bull_done and all(lows[i - k] >= ol for k in range(HOLD_BARS))
                    and closes[i] >= mid
                    and ((not require_breakout_side) or closes[i] >= oh)):
                records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                                "entry_price": float(closes[i])})
                bull_done = True
            if (not bear_done and all(highs[i - k] <= oh for k in range(HOLD_BARS))
                    and closes[i] <= mid
                    and ((not require_breakout_side) or closes[i] <= ol)):
                records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                                "entry_price": float(closes[i])})
                bear_done = True
    return pd.DataFrame(records)


def build_paths(entries):
    """For each entry, list of (delta, time) after entry until hard exit, using bar highs/lows for path."""
    paths = []
    meta = []
    for _, row in entries.iterrows():
        ep = float(row["entry_price"])
        et = row["entry_time"]
        direction = row["direction"]
        day = row["day"]
        day_bars = df[(df.index.date == day) & (df.index > et) & (df.index.time <= HARD_EXIT)]
        deltas = []
        times = []
        for ts, bar in day_bars.iterrows():
            # use close for decision (matches prior scripts); also track extreme in direction
            px = float(bar["close"])
            dlt = (px - ep) if direction == "LONG" else (ep - px)
            # favourable extreme within bar (optimistic target hit)
            if direction == "LONG":
                fav = float(bar["high"]) - ep
                adv = ep - float(bar["low"])
            else:
                fav = ep - float(bar["low"])
                adv = float(bar["high"]) - ep
            deltas.append((dlt, fav, adv, ts.time()))
            times.append(ts)
        paths.append(deltas)
        meta.append({"day": day, "direction": direction, "entry_time": et, "entry_price": ep})
    return paths, meta


def score_path(path, body, debit, stop_pts):
    """Return (spot_delta, reason) using fav for target, adv for stop, close for hard."""
    if not path:
        return None, None
    for dlt, fav, adv, t in path:
        if fav >= body:
            return float(body), "TARGET"
        if adv >= stop_pts:
            return float(-stop_pts), "STOP"
        if t >= HARD_EXIT:
            return float(dlt), "HARD_EXIT"
    dlt, fav, adv, t = path[-1]
    return float(dlt), "HARD_EXIT"


signal_sets = {
    "hold_base": generate_hold_signals(False, require_breakout_side=False),
    "hold_quiet": generate_hold_signals(True, require_breakout_side=False),
    "hold_break": generate_hold_signals(False, require_breakout_side=True),
    "hold_quiet_break": generate_hold_signals(True, require_breakout_side=True),
}
for k, v in signal_sets.items():
    print(f"Signals {k}: {len(v)}", flush=True)

precomputed = {}
for name, entries in signal_sets.items():
    if entries.empty:
        continue
    print(f"Building paths for {name}...", flush=True)
    precomputed[name] = build_paths(entries)

BODIES = [50, 80, 100]
FAR_EXTRA = [50, 100]
COST_PCTS = [0.15, 0.20, 0.30]
STOP_MULTS = [0.5, 0.75, 1.0]
# Also fixed stops independent of debit
FIXED_STOPS = [15, 20, 25]

results = []
best = None
best_detail = None

for sig_name, (paths, meta) in precomputed.items():
    configs = []
    for body, far_extra, cost_pct, stop_mult in product(BODIES, FAR_EXTRA, COST_PCTS, STOP_MULTS):
        far = body + far_extra
        narrow = min(body, far - body)
        debit = narrow * cost_pct
        stop_pts = max(10.0, stop_mult * debit)
        configs.append((body, far, debit, stop_pts, cost_pct, stop_mult, "mult"))
    for body, far_extra, cost_pct, stop_pts in product(BODIES, FAR_EXTRA, COST_PCTS, FIXED_STOPS):
        far = body + far_extra
        narrow = min(body, far - body)
        debit = narrow * cost_pct
        configs.append((body, far, debit, float(stop_pts), cost_pct, None, "fixed"))

    for body, far, debit, stop_pts, cost_pct, stop_mult, stop_mode in configs:
        days, pnls, reasons = [], [], []
        for path, m in zip(paths, meta):
            delta, reason = score_path(path, body, debit, stop_pts)
            if delta is None:
                continue
            pnl = butterfly_intrinsic(delta, body, far) - debit
            days.append(m["day"])
            pnls.append(pnl)
            reasons.append(reason)
        if len(pnls) < 30:
            continue
        met = metrics_from_pnl(days, pnls)
        row = {
            "signals": sig_name, "body": body, "far": far,
            "wings": f"{body}/{far - body}", "cost_pct": cost_pct, "debit": debit,
            "stop_pts": stop_pts, "stop_mode": stop_mode, "stop_mult": stop_mult,
            **met,
            "n_target": reasons.count("TARGET"),
            "n_stop": reasons.count("STOP"),
            "n_hard": reasons.count("HARD_EXIT"),
            "pass_gate": (met["sharpe"] > SHARPE_GATE) and (met["wr"] >= WR_GATE),
        }
        results.append(row)
        if row["pass_gate"]:
            key = (met["sharpe"], met["wr"], met["pf"])
            if best is None or key > best:
                best = key
                best_detail = (row, paths, meta, body, far, debit, stop_pts)

res_df = pd.DataFrame(results).sort_values(
    ["pass_gate", "sharpe", "wr"], ascending=[False, False, False]
)
out_csv = os.path.join(OUT_DIR, "orb_asym_v2_search.csv")
res_df.to_csv(out_csv, index=False)

print("\n" + "=" * 100, flush=True)
print(f"SEARCH n={len(res_df)}  gates Sharpe>{SHARPE_GATE} WR>={WR_GATE}%", flush=True)
print(res_df.head(20).to_string(index=False), flush=True)
passed = res_df[res_df["pass_gate"]]
print(f"\nPassing: {len(passed)}", flush=True)

if best_detail is not None:
    row, paths, meta, body, far, debit, stop_pts = best_detail
    print("\n*** BEST PASSING ***", flush=True)
    print(row, flush=True)
    trades = []
    for path, m in zip(paths, meta):
        delta, reason = score_path(path, body, debit, stop_pts)
        if delta is None:
            continue
        trades.append({
            **m, "spot_delta": delta, "reason": reason,
            "bf_pnl": butterfly_intrinsic(delta, body, far) - debit,
            "body": body, "far": far, "debit": debit, "stop_pts": stop_pts,
        })
    tdf = pd.DataFrame(trades)
    tpath = os.path.join(OUT_DIR, "orb_asym_v2_best_trades.csv")
    tdf.to_csv(tpath, index=False)
    print(f"trades -> {tpath}", flush=True)
else:
    print("\nNo passer. Top 5 by Sharpe:", flush=True)
    print(res_df.head(5).to_string(index=False), flush=True)

print(f"\nFull -> {out_csv}", flush=True)
print("Done.", flush=True)
