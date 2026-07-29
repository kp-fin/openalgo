"""
Nifty Next 50 Cash Basket — Backtest
Universe : Top 10 Nifty Next 50 by ADV, static current constituents
           (same caveat as EMA Regime Crossover backtest — today's ranking
           held constant across the window; TMCV / MUTHOOTFIN may have
           limited history)
Adani cap: max 2 stocks from any single conglomerate → ADANIGREEN excluded
           (3rd Adani by ADV); 10th slot filled by MUTHOOTFIN
Entry     : First trading day of each month, at open price
Target    : +10% from entry (intraday high touch)
Stop      : entry − ATR_MULTIPLIER × ATR(ATR_PERIOD, daily) at time of entry
            (intraday low touch)
Post-exit : Cash held until next monthly rebalance
Rebalance : 1st trading day of each month — exit names no longer in
            universe, enter new names
Sizing    : Rs 5,00,000 total / 10 equal slots = Rs 50,000 per slot
            qty = floor(50_000 / entry_open)
Product   : CNC delivery (no leverage)
Window    : 2021-07-01 → 2026-06-30
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"
IST        = ZoneInfo("Asia/Kolkata")

TOTAL_CAPITAL    = 500_000
N_SLOTS          = 10
CAPITAL_PER_SLOT = TOTAL_CAPITAL // N_SLOTS   # 50,000

TARGET_PCT      = 0.10    # +10% from entry
ATR_PERIOD      = 14
ATR_MULTIPLIER  = 1.5     # stop = entry − 1.5 × ATR(14, daily)

# Static universe — Adani cap applied (max 2 per group)
# Ranking by trailing 20-day ADV as of 2026-07-17 (universe_ranking.py)
# ADANIGREEN excluded (3rd Adani); MUTHOOTFIN fills the 10th slot
UNIVERSE = [
    "ADANIPOWER",   # Adani #1 by ADV
    "VEDL",
    "CGPOWER",
    "LODHA",
    "ADANIENSOL",   # Adani #2 by ADV  (ADANIGREEN excluded — group cap)
    "TORNTPHARM",
    "TVSMOTOR",
    "CANBK",
    "MUTHOOTFIN",
    "TMCV",
]

# ── Client ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)

# ── Data fetch ───────────────────────────────────────────────────────────────
def fetch_daily(symbol: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        resp = client.history(
            symbol=symbol, exchange="NSE", interval="D",
            start_date=START_DATE, end_date=END_DATE
        )
    except Exception as e:
        return None, f"error: {e}"

    if isinstance(resp, dict):
        if resp.get("status") != "success":
            return None, f"api error: {resp.get('message', resp)}"
        df = pd.DataFrame(resp.get("data", []))
        if df.empty:
            return None, "no data"
        df.columns = [c.lower() for c in df.columns]
        # dict response has a datetime/timestamp column to parse
        dt_col = next((c for c in df.columns if c in ("datetime", "timestamp", "date")), df.columns[0])
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()
    elif isinstance(resp, pd.DataFrame):
        # SDK already returns a DataFrame with timestamp as index
        df = resp.copy()
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    else:
        return None, "unexpected response type"

    if df is None or df.empty:
        return None, "no data"

    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert(IST)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to requested window
    start = pd.Timestamp(START_DATE, tz="Asia/Kolkata")
    end   = pd.Timestamp(END_DATE,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
    df = df.loc[start:end]

    return df, None

# ── ATR (Wilder's smoothing) ─────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ── Rebalance calendar ───────────────────────────────────────────────────────
def build_rebalance_dates(all_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    months = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    rebal = []
    for m in months:
        bucket = all_dates[(all_dates.year == m.year) & (all_dates.month == m.month)]
        if len(bucket):
            rebal.append(bucket[0])
    return sorted(set(rebal))

# ── Backtest engine ──────────────────────────────────────────────────────────
def run_backtest(price_data: dict[str, pd.DataFrame]) -> dict:
    all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    rebal_set = set(build_rebalance_dates(all_dates))

    active = {}     # symbol → {entry_price, qty, stop, target, entry_date, atr}
    cash   = float(TOTAL_CAPITAL)
    trades = []
    equity = []

    def close_pos(sym, price, date, reason):
        nonlocal cash
        pos = active.pop(sym)
        pnl = (price - pos["entry_price"]) * pos["qty"]
        cash += price * pos["qty"]
        trades.append({
            "symbol"      : sym,
            "entry_date"  : pos["entry_date"].date(),
            "exit_date"   : date.date(),
            "entry_price" : round(pos["entry_price"], 2),
            "exit_price"  : round(price, 2),
            "qty"         : pos["qty"],
            "stop"        : round(pos["stop"], 2),
            "target"      : round(pos["target"], 2),
            "atr_at_entry": round(pos["atr"], 2),
            "pnl"         : round(pnl, 2),
            "reason"      : reason,
            "holding_days": (date - pos["entry_date"]).days,
        })

    def open_pos(sym, date, df_sym):
        nonlocal cash
        if date not in df_sym.index:
            return
        bar = df_sym.loc[date]
        entry = bar["open"]
        if not entry or entry <= 0:
            return

        hist = df_sym.loc[:date]
        if len(hist) < ATR_PERIOD + 2:
            return
        atr_val = compute_atr(hist, ATR_PERIOD).iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return

        stop   = entry - ATR_MULTIPLIER * atr_val
        target = entry * (1 + TARGET_PCT)
        qty    = int(CAPITAL_PER_SLOT // entry)
        if qty <= 0:
            return
        cost = qty * entry
        if cost > cash:
            return

        cash -= cost
        active[sym] = {
            "entry_price": entry,
            "qty"        : qty,
            "stop"       : stop,
            "target"     : target,
            "entry_date" : date,
            "atr"        : atr_val,
        }

    for date in all_dates:
        # 1. Check stop / target for open positions
        for sym in list(active.keys()):
            if sym not in price_data or date not in price_data[sym].index:
                continue
            bar = price_data[sym].loc[date]
            pos = active[sym]
            if date == pos["entry_date"]:
                continue   # don't check stop/target on entry bar
            if bar["low"] <= pos["stop"]:
                close_pos(sym, pos["stop"], date, "STOP")
            elif bar["high"] >= pos["target"]:
                close_pos(sym, pos["target"], date, "TARGET")

        # 2. Monthly rebalance
        if date in rebal_set:
            desired = set(s for s in UNIVERSE if s in price_data)
            current = set(active.keys())

            # Exit stale holdings
            for sym in current - desired:
                if sym in price_data and date in price_data[sym].index:
                    close_pos(sym, price_data[sym].loc[date]["open"], date, "REBALANCE_EXIT")

            # Enter missing slots
            for sym in desired - set(active.keys()):
                open_pos(sym, date, price_data[sym])

        # 3. Mark-to-market equity
        mtm = cash
        for sym, pos in active.items():
            if sym in price_data and date in price_data[sym].index:
                mtm += price_data[sym].loc[date]["close"] * pos["qty"]
        equity.append({"date": date.date(), "equity": round(mtm, 2)})

    # Close remaining at end
    last = all_dates[-1]
    for sym in list(active.keys()):
        if sym in price_data and last in price_data[sym].index:
            close_pos(sym, price_data[sym].loc[last]["close"], last, "END_OF_BACKTEST")

    return {
        "trades"      : pd.DataFrame(trades),
        "equity_curve": pd.DataFrame(equity),
    }

# ── Metrics ──────────────────────────────────────────────────────────────────
def print_metrics(results: dict) -> None:
    trades = results["trades"]
    equity = results["equity_curve"]

    if trades.empty:
        print("No trades executed.")
        return

    wins   = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    wr     = len(wins) / len(trades) * 100
    gp     = wins["pnl"].sum()
    gl     = losses["pnl"].abs().sum()
    pf     = gp / gl if gl > 0 else float("inf")
    net    = trades["pnl"].sum()

    eq     = equity.set_index("date")["equity"]
    dd     = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    dret   = eq.pct_change().dropna()
    sharpe = (dret.mean() / dret.std() * np.sqrt(252)) if dret.std() > 0 else 0
    n_yr   = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25
    cagr   = ((eq.iloc[-1] / TOTAL_CAPITAL) ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0

    W = 58
    print("\n" + "=" * W)
    print("  NIFTY NEXT 50 BASKET — BACKTEST RESULTS")
    print("=" * W)
    print(f"  Period       : {START_DATE}  →  {END_DATE}")
    print(f"  Universe     : {len(UNIVERSE)} stocks  |  Adani cap: 2")
    print(f"  Target / Stop: +{TARGET_PCT*100:.0f}%  /  {ATR_MULTIPLIER}×ATR({ATR_PERIOD}, daily)")
    print(f"  Capital      : ₹{TOTAL_CAPITAL:,.0f}  ({N_SLOTS} slots × ₹{CAPITAL_PER_SLOT:,.0f})")
    print("-" * W)
    print(f"  Total trades : {len(trades)}")
    print(f"  Win rate     : {wr:.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Net P&L      : ₹{net:,.0f}")
    print(f"  CAGR         : {cagr:.1f}%")
    print(f"  Max drawdown : {dd:.1f}%")
    print(f"  Sharpe ratio : {sharpe:.2f}")
    print(f"  Avg win      : ₹{wins['pnl'].mean():,.0f}")
    print(f"  Avg loss     : ₹{losses['pnl'].mean():,.0f}")
    print(f"  Avg holding  : {trades['holding_days'].mean():.0f} days")
    print("-" * W)
    print("  Exit breakdown:")
    for reason, cnt in trades["reason"].value_counts().items():
        print(f"    {reason:<22}: {cnt}")
    print("-" * W)
    print("  Per-symbol P&L (sorted best → worst):")
    sym = trades.groupby("symbol")["pnl"].agg(["sum", "count", "mean"])
    sym.columns = ["total", "n", "avg"]
    for s, r in sym.sort_values("total", ascending=False).iterrows():
        bar = "+" if r["total"] >= 0 else ""
        print(f"    {s:<15}: {bar}₹{r['total']:>9,.0f}  ({int(r['n'])} trades, avg {bar}₹{r['avg']:,.0f})")
    print("=" * W)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Nifty Next 50 Basket Backtest  |  {START_DATE} → {END_DATE}")
    print(f"Target +{TARGET_PCT*100:.0f}%  |  Stop {ATR_MULTIPLIER}×ATR({ATR_PERIOD})  |  ₹{CAPITAL_PER_SLOT:,.0f}/slot\n")

    price_data: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        print(f"  Fetching {sym:<15}...", end=" ", flush=True)
        df, err = fetch_daily(sym)
        if err:
            print(f"SKIP ({err})")
        else:
            print(f"{len(df)} daily bars  ({df.index[0].date()} → {df.index[-1].date()})")
            price_data[sym] = df

    print(f"\nRunning backtest on {len(price_data)} symbols...")
    results = run_backtest(price_data)
    print_metrics(results)

    out_dir = os.path.join(os.path.dirname(__file__), "nifty_next50_basket")
    os.makedirs(out_dir, exist_ok=True)
    results["trades"].to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    results["equity_curve"].to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)
    print(f"\n  Trade log    → {out_dir}/trades.csv")
    print(f"  Equity curve → {out_dir}/equity_curve.csv")
