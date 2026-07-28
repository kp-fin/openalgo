"""
ORB_Spread — Spot Signals, Futures P&L Exit Test
Nifty 50 | 2021-07-01 → 2026-06-30

Hypothesis (Option 2): keep spot data for OR construction and signal
detection (unchanged from live script), but measure exit P&L against
Nifty monthly futures movement instead of spot movement.

Rationale: options are priced off futures, not the index. If the futures
price moves differently from spot during a trade (basis compression/
expansion, expiry day distortions), the TARGET/STOP triggers may fire at
different times — changing which trades win, lose, or time-out.

Method:
  1. Spot 5m (NSE_INDEX) — OR levels, signal detection, range-day check
     (identical to existing backtests).
  2. Futures 5m (NFO, near-month contract) — stitched continuous series
     built by fetching each monthly contract and rolling on expiry day.
     Used ONLY for entry_futures price and exit P&L tracking.
  3. Fallback: if a contract's data is unavailable (not in local DuckDB
     cache), that day's exits fall back to spot — logged clearly.

Output: side-by-side table (spot-based vs futures-based) + CSVs.
"""

import calendar
import os
import sys
import time
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytz
import requests

warnings.filterwarnings("ignore")

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"

OR_MIN, OR_MAX       = 30, 150
ENTRY_END_H, ENTRY_END_M = 12, 0
HARD_EXIT_H, HARD_EXIT_M = 15, 15
RANGE_CHK_H, RANGE_CHK_M = 10, 15
TARGET_PTS, STOP_PTS  = 40, 25
PREV_MOVE_THRESHOLD   = 0.42

IST    = pytz.timezone("Asia/Kolkata")
HEADERS = {"Content-Type": "application/json"}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orb_v2_futures_exit")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_history(symbol, exchange, interval, start, end, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{HOST}/api/v1/history",
                json={"apikey": API_KEY, "symbol": symbol, "exchange": exchange,
                      "interval": interval, "start_date": start, "end_date": end},
                headers=HEADERS, timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "success" or not data.get("data"):
                return pd.DataFrame()
            df = pd.DataFrame(data["data"])
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST).dt.tz_localize(None)
                df = df.set_index("datetime")
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close"]].sort_index()
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def last_thursday(year, month):
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def build_contract_schedule(start_str, end_str):
    """Return list of (month_start, expiry_date, symbol) for each monthly contract."""
    start = date.fromisoformat(start_str)
    end   = date.fromisoformat(end_str)
    contracts = []
    d = date(start.year, start.month, 1)
    while d <= end:
        exp = last_thursday(d.year, d.month)
        sym = f"NIFTY{exp.strftime('%d%b%y').upper()}FUT"
        contracts.append((d, exp, sym))
        d = date(d.year + (d.month // 12), (d.month % 12) + 1, 1)
    return contracts


def build_futures_series(contracts):
    """
    Fetch each monthly futures contract and stitch into one continuous 5m DataFrame.
    A contract is active from the day after the previous expiry through its own expiry.
    Falls back gracefully — days with no futures data are excluded from the futures index.
    """
    frames = []
    prev_expiry = None

    for i, (month_start, expiry, sym) in enumerate(contracts):
        # Active window: day after previous contract's expiry (or START_DATE) to this expiry
        if prev_expiry is None:
            fetch_start = START_DATE
        else:
            fetch_start = (prev_expiry + timedelta(days=1)).isoformat()
        fetch_end = expiry.isoformat()

        print(f"  Fetching {sym}  ({fetch_start} → {fetch_end})", end="", flush=True)
        df = fetch_history(sym, "NFO", "5m", fetch_start, fetch_end)
        if df.empty:
            print(f"  [NO DATA — days in this window will fall back to spot]")
        else:
            print(f"  {len(df):,} bars")
            frames.append(df)

        prev_expiry = expiry
        time.sleep(0.3)   # polite rate-limit

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ── Spot data ─────────────────────────────────────────────────────────────────
print("Fetching Nifty spot 5m data …")
spot_df = fetch_history("NIFTY", "NSE_INDEX", "5m", START_DATE, END_DATE)
if spot_df.empty:
    raise SystemExit("Spot data fetch failed.")
print(f"Spot: {len(spot_df):,} bars  {spot_df.index[0]} → {spot_df.index[-1]}")

# ── Previous-day net move (spot, same as live) ────────────────────────────────
daily_spot = spot_df.groupby(spot_df.index.date).agg(
    open=("open", "first"), close=("close", "last")
)
daily_spot["net_move_pct"]  = (daily_spot["close"] - daily_spot["open"]).abs() / daily_spot["open"] * 100
daily_spot["prev_net_move"] = daily_spot["net_move_pct"].shift(1)

# ── Futures continuous series ─────────────────────────────────────────────────
print("\nBuilding stitched Nifty futures series …")
contracts = build_contract_schedule(START_DATE, END_DATE)
futures_df = build_futures_series(contracts)
if futures_df.empty:
    print("WARNING: No futures data returned at all — futures-based column will be all-spot fallback.")
else:
    print(f"\nFutures stitched: {len(futures_df):,} bars  {futures_df.index[0]} → {futures_df.index[-1]}")

# Index futures by date for fast per-day lookup
futures_by_date = {}
if not futures_df.empty:
    for d, grp in futures_df.groupby(futures_df.index.date):
        futures_by_date[d] = grp


# ── Signal generation (spot only — identical to existing backtests) ───────────
print("\nGenerating signals from spot data …")
records = []

for day, grp in spot_df.groupby(spot_df.index.date):
    prev_net = daily_spot["prev_net_move"].get(day, np.nan)
    if np.isnan(prev_net):
        continue   # first day, no prior-day reading

    or_window = grp.between_time("09:15", "09:44")
    if len(or_window) < 3:
        continue
    orb_high = or_window["high"].max()
    orb_low  = or_window["low"].min()
    if not (OR_MIN <= orb_high - orb_low <= OR_MAX):
        continue

    # Range-day check at 10:15 (spot close vs OR levels)
    at_1015 = grp.between_time("10:15", "10:15")
    if not at_1015.empty and orb_low < at_1015["close"].iloc[0] < orb_high:
        continue

    # Previous-day filter
    if prev_net > PREV_MOVE_THRESHOLD:
        continue

    sig = grp.between_time("09:45", "12:00")
    if len(sig) < 2:
        continue

    c  = sig["close"].values
    o  = sig["open"].values
    h  = sig["high"].values
    lo = sig["low"].values
    idx = sig.index
    bear_done = bull_done = False

    for i in range(1, len(sig)):
        t = idx[i].time()
        if t.hour > ENTRY_END_H or (t.hour == ENTRY_END_H and t.minute >= ENTRY_END_M):
            break

        if not bear_done and c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                            "signal": "BearishReject", "entry_spot": c[i],
                            "orb_high": orb_high, "orb_low": orb_low})
            bear_done = True
        if not bull_done and c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                            "signal": "BullishReject", "entry_spot": c[i],
                            "orb_high": orb_high, "orb_low": orb_low})
            bull_done = True
        if not bear_done and i >= 2 and h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
            records.append({"day": day, "entry_time": idx[i], "direction": "SHORT",
                            "signal": "LowerHigh", "entry_spot": c[i],
                            "orb_high": orb_high, "orb_low": orb_low})
            bear_done = True
        if not bull_done and i >= 2 and lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
            records.append({"day": day, "entry_time": idx[i], "direction": "LONG",
                            "signal": "HigherLow", "entry_spot": c[i],
                            "orb_high": orb_high, "orb_low": orb_low})
            bull_done = True

entries_df = pd.DataFrame(records).reset_index(drop=True)
print(f"Signals: {len(entries_df)}")

# ── Attach futures entry price ────────────────────────────────────────────────
def get_futures_price_at(day, ts):
    """Return futures close at timestamp ts, or None if unavailable."""
    day_futures = futures_by_date.get(day)
    if day_futures is None or day_futures.empty:
        return None
    # find the bar whose index == ts (exact), fall back to nearest before
    exact = day_futures[day_futures.index == ts]
    if not exact.empty:
        return float(exact["close"].iloc[0])
    before = day_futures[day_futures.index <= ts]
    if not before.empty:
        return float(before["close"].iloc[-1])
    return None

entries_df["entry_futures"] = entries_df.apply(
    lambda r: get_futures_price_at(r["day"], r["entry_time"]), axis=1
)
n_no_futures = entries_df["entry_futures"].isna().sum()
print(f"Entries with futures price: {entries_df['entry_futures'].notna().sum()} / {len(entries_df)} "
      f"({n_no_futures} will use spot fallback for futures column)")


# ── Simulate exits ────────────────────────────────────────────────────────────
def simulate_exits(entries, price_df, use_col):
    """
    Simulate exits using price_df[use_col] for P&L tracking.
    use_col: 'spot' or 'futures' — determines which series drives TARGET/STOP.
    For 'futures', falls back to spot on days with no futures data.
    """
    out = entries.copy()
    out["exit_time"] = pd.NaT
    out["exit_price"] = np.nan
    out["pnl_pts"] = np.nan
    out["reason"] = ""
    out["data_source"] = ""

    for row_i, row in out.iterrows():
        day = row["day"]
        sign = 1 if row["direction"] == "SHORT" else -1

        if use_col == "futures":
            day_price = futures_by_date.get(day)
            if day_price is None or day_price.empty:
                # Fallback to spot for this day
                day_price = spot_df[spot_df.index.date == day]
                entry_px  = row["entry_spot"]
                source    = "spot_fallback"
            else:
                entry_px = row["entry_futures"]
                if pd.isna(entry_px):
                    day_price = spot_df[spot_df.index.date == day]
                    entry_px  = row["entry_spot"]
                    source    = "spot_fallback"
                else:
                    source = "futures"
        else:
            day_price = spot_df[spot_df.index.date == day]
            entry_px  = row["entry_spot"]
            source    = "spot"

        bars_after = day_price[day_price.index > row["entry_time"]]

        for ts, bar in bars_after.iterrows():
            px  = float(bar["close"])
            pnl = (entry_px - px) * sign
            t   = ts.time()
            hard = t.hour > HARD_EXIT_H or (t.hour == HARD_EXIT_H and t.minute >= HARD_EXIT_M)

            if pnl >= TARGET_PTS:
                reason = "TARGET"
            elif pnl <= -STOP_PTS:
                reason = "STOP"
            elif hard:
                reason = "HARD_EXIT"
            else:
                continue

            out.at[row_i, "exit_time"]   = ts
            out.at[row_i, "exit_price"]  = px
            out.at[row_i, "pnl_pts"]     = pnl
            out.at[row_i, "reason"]      = reason
            out.at[row_i, "data_source"] = source
            break

    return out.dropna(subset=["pnl_pts"]).copy()


print("\nSimulating exits …")
spot_trades    = simulate_exits(entries_df, spot_df,    use_col="spot")
futures_trades = simulate_exits(entries_df, futures_df, use_col="futures")

spot_trades.to_csv(os.path.join(OUT_DIR, "spot_trades.csv"), index=False)
futures_trades.to_csv(os.path.join(OUT_DIR, "futures_trades.csv"), index=False)

# How many futures trades actually used futures vs fell back
if not futures_trades.empty:
    src_counts = futures_trades["data_source"].value_counts().to_dict()
else:
    src_counts = {}


# ── Summarise ─────────────────────────────────────────────────────────────────
def summarise(df, label):
    if df.empty:
        print(f"\n{label}: no trades")
        return {}
    n   = len(df)
    wr  = (df["pnl_pts"] > 0).mean() * 100
    avg = df["pnl_pts"].mean()
    gw  = df[df["pnl_pts"] > 0]["pnl_pts"].sum()
    gl  = abs(df[df["pnl_pts"] <= 0]["pnl_pts"].sum())
    pf  = gw / gl if gl > 0 else float("inf")
    exits = df["reason"].value_counts().to_dict()
    print(f"\n── {label} {'─'*(60-len(label))}")
    print(f"  Trades        : {n}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Avg P&L       : {avg:+.2f} pts")
    print(f"  Total P&L     : {df['pnl_pts'].sum():+.1f} pts")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Exits         : {exits}")
    return {"n": n, "wr": wr, "avg": avg, "total": df["pnl_pts"].sum(), "pf": pf}


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
res_spot    = summarise(spot_trades,    "Spot-based exit (baseline)")
res_futures = summarise(futures_trades, "Futures-based exit (option 2)")

if src_counts:
    print(f"\n  Futures data coverage: {src_counts}")

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print(f"  {'Metric':<20} {'Spot':>12} {'Futures':>12} {'Delta':>12}")
print("  " + "-" * 58)
for metric, fmt in [("Trades", "{:>12d}"), ("Win rate %", "{:>12.1f}"),
                    ("Avg P&L pts", "{:>12.2f}"), ("Total P&L pts", "{:>12.1f}"),
                    ("Profit factor", "{:>12.2f}")]:
    key = {"Trades": "n", "Win rate %": "wr", "Avg P&L pts": "avg",
           "Total P&L pts": "total", "Profit factor": "pf"}[metric]
    sv = res_spot.get(key, 0)
    fv = res_futures.get(key, 0)
    delta = fv - sv
    print(f"  {metric:<20}" + fmt.format(sv) + fmt.format(fv) + fmt.format(delta))

# ── Trades that changed outcome (TARGET<->STOP<->HARD_EXIT) ──────────────────
if not spot_trades.empty and not futures_trades.empty:
    merged = spot_trades[["day", "entry_time", "direction", "signal", "reason", "pnl_pts"]].merge(
        futures_trades[["day", "entry_time", "direction", "reason", "pnl_pts", "data_source"]],
        on=["day", "entry_time", "direction"], suffixes=("_spot", "_futures"),
    )
    changed = merged[merged["reason_spot"] != merged["reason_futures"]]
    print(f"\n  Trades where exit reason changed (spot→futures): {len(changed)}")
    if not changed.empty:
        print(changed[["day", "direction", "signal", "reason_spot", "pnl_pts_spot",
                        "reason_futures", "pnl_pts_futures", "data_source"]].to_string(index=False))
    changed.to_csv(os.path.join(OUT_DIR, "outcome_changes.csv"), index=False)

print(f"\nCSVs saved to: {OUT_DIR}")
print("Backtest complete.")
