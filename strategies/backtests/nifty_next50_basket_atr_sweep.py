"""
Nifty Next 50 Basket — ATR Multiplier Sweep
Tests ATR_MULTIPLIER in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
All other params fixed: +10% target, Rs 50,000/slot, 10 slots, 2021-07-01→2026-06-30
Reuses fetch + engine from nifty_next50_basket_backtest.py logic.
"""

import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

START_DATE = "2021-07-01"
END_DATE   = "2026-06-30"
IST        = ZoneInfo("Asia/Kolkata")

TOTAL_CAPITAL    = 500_000
N_SLOTS          = 10
CAPITAL_PER_SLOT = TOTAL_CAPITAL // N_SLOTS

TARGET_PCT = 0.10
ATR_PERIOD = 14

UNIVERSE = [
    "ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
    "TORNTPHARM", "TVSMOTOR", "CANBK", "MUTHOOTFIN", "TMCV",
]

ATR_SWEEP = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

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
        dt_col = next((c for c in df.columns if c in ("datetime", "timestamp", "date")), df.columns[0])
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()
    elif isinstance(resp, pd.DataFrame):
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

    start = pd.Timestamp(START_DATE, tz="Asia/Kolkata")
    end   = pd.Timestamp(END_DATE,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
    return df.loc[start:end], None

# ── ATR ──────────────────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ── Rebalance calendar ───────────────────────────────────────────────────────
def build_rebalance_dates(all_dates: pd.DatetimeIndex) -> set:
    months = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    out = set()
    for m in months:
        bucket = all_dates[(all_dates.year == m.year) & (all_dates.month == m.month)]
        if len(bucket):
            out.add(bucket[0])
    return out

# ── Backtest engine ──────────────────────────────────────────────────────────
def run_backtest(price_data: dict, atr_mult: float) -> dict:
    all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
    all_dates  = pd.DatetimeIndex(all_dates)
    rebal_set  = build_rebalance_dates(all_dates)

    active = {}
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
            "entry_price" : pos["entry_price"],
            "exit_price"  : price,
            "qty"         : pos["qty"],
            "stop"        : pos["stop"],
            "target"      : pos["target"],
            "pnl"         : round(pnl, 2),
            "reason"      : reason,
            "holding_days": (date - pos["entry_date"]).days,
        })

    def open_pos(sym, date):
        nonlocal cash
        df_sym = price_data[sym]
        if date not in df_sym.index:
            return
        bar   = df_sym.loc[date]
        entry = bar["open"]
        if not entry or entry <= 0:
            return
        hist = df_sym.loc[:date]
        if len(hist) < ATR_PERIOD + 2:
            return
        atr_val = compute_atr(hist, ATR_PERIOD).iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return
        stop   = entry - atr_mult * atr_val
        target = entry * (1 + TARGET_PCT)
        qty    = int(CAPITAL_PER_SLOT // entry)
        if qty <= 0:
            return
        cost = qty * entry
        if cost > cash:
            return
        cash -= cost
        active[sym] = {
            "entry_price": entry, "qty": qty,
            "stop": stop, "target": target, "entry_date": date,
        }

    for date in all_dates:
        for sym in list(active.keys()):
            if sym not in price_data or date not in price_data[sym].index:
                continue
            bar = price_data[sym].loc[date]
            pos = active[sym]
            if date == pos["entry_date"]:
                continue
            if bar["low"] <= pos["stop"]:
                close_pos(sym, pos["stop"], date, "STOP")
            elif bar["high"] >= pos["target"]:
                close_pos(sym, pos["target"], date, "TARGET")

        if date in rebal_set:
            desired = set(s for s in UNIVERSE if s in price_data)
            for sym in set(active.keys()) - desired:
                if sym in price_data and date in price_data[sym].index:
                    close_pos(sym, price_data[sym].loc[date]["open"], date, "REBALANCE_EXIT")
            for sym in desired - set(active.keys()):
                open_pos(sym, date)

        mtm = cash
        for sym, pos in active.items():
            if sym in price_data and date in price_data[sym].index:
                mtm += price_data[sym].loc[date]["close"] * pos["qty"]
        equity.append({"date": date.date(), "equity": round(mtm, 2)})

    last = all_dates[-1]
    for sym in list(active.keys()):
        if sym in price_data and last in price_data[sym].index:
            close_pos(sym, price_data[sym].loc[last]["close"], last, "END_OF_BACKTEST")

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)

    if trades_df.empty:
        return {"atr_mult": atr_mult, "trades": 0}

    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr     = len(wins) / len(trades_df) * 100
    gp     = wins["pnl"].sum()
    gl     = losses["pnl"].abs().sum()
    pf     = gp / gl if gl > 0 else float("inf")
    net    = trades_df["pnl"].sum()

    eq    = equity_df.set_index("date")["equity"]
    dd    = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    dret  = eq.pct_change().dropna()
    sharpe = (dret.mean() / dret.std() * np.sqrt(252)) if dret.std() > 0 else 0
    n_yr  = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days / 365.25
    cagr  = ((eq.iloc[-1] / TOTAL_CAPITAL) ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0

    stops   = (trades_df["reason"] == "STOP").sum()
    targets = (trades_df["reason"] == "TARGET").sum()

    return {
        "atr_mult"   : atr_mult,
        "trades"     : len(trades_df),
        "win_rate"   : round(wr, 1),
        "pf"         : round(pf, 2),
        "net_pnl"    : round(net, 0),
        "cagr"       : round(cagr, 1),
        "max_dd"     : round(dd, 1),
        "sharpe"     : round(sharpe, 2),
        "avg_win"    : round(wins["pnl"].mean(), 0) if len(wins) else 0,
        "avg_loss"   : round(losses["pnl"].mean(), 0) if len(losses) else 0,
        "avg_hold"   : round(trades_df["holding_days"].mean(), 0),
        "stops"      : stops,
        "targets"    : targets,
        "trades_df"  : trades_df,
    }

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Nifty Next 50 Basket — ATR Multiplier Sweep")
    print(f"Target +{TARGET_PCT*100:.0f}%  |  ATR period {ATR_PERIOD}  |  ₹{CAPITAL_PER_SLOT:,.0f}/slot\n")

    print("Fetching daily data (once, reused across all sweeps)...")
    price_data: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        print(f"  {sym:<15}...", end=" ", flush=True)
        df, err = fetch_daily(sym)
        if err:
            print(f"SKIP ({err})")
        else:
            print(f"{len(df)} bars")
            price_data[sym] = df

    print(f"\nRunning sweep: {ATR_SWEEP}\n")
    rows = []
    for mult in ATR_SWEEP:
        print(f"  ATR×{mult:.2f}...", end=" ", flush=True)
        r = run_backtest(price_data, mult)
        rows.append(r)
        print(f"WR {r.get('win_rate','?')}%  PF {r.get('pf','?')}  CAGR {r.get('cagr','?')}%  DD {r.get('max_dd','?')}%  Sharpe {r.get('sharpe','?')}")

    W = 108
    print("\n" + "=" * W)
    print(f"{'Mult':>6}  {'Trades':>7}  {'WR%':>5}  {'PF':>5}  {'Net P&L':>12}  {'CAGR%':>6}  {'Max DD%':>7}  {'Sharpe':>7}  {'Avg Win':>9}  {'Avg Loss':>9}  {'AvgHold':>8}  {'Stops':>6}  {'Targets':>8}")
    print("-" * W)
    for r in rows:
        if r.get("trades", 0) == 0:
            print(f"  {r['atr_mult']:>4.2f}  NO TRADES")
            continue
        print(
            f"  {r['atr_mult']:>4.2f}  "
            f"{r['trades']:>7}  "
            f"{r['win_rate']:>5.1f}  "
            f"{r['pf']:>5.2f}  "
            f"₹{r['net_pnl']:>10,.0f}  "
            f"{r['cagr']:>6.1f}  "
            f"{r['max_dd']:>7.1f}  "
            f"{r['sharpe']:>7.2f}  "
            f"₹{r['avg_win']:>8,.0f}  "
            f"₹{r['avg_loss']:>8,.0f}  "
            f"{r['avg_hold']:>8.0f}  "
            f"{r['stops']:>6}  "
            f"{r['targets']:>8}"
        )
    print("=" * W)

    # Save summary
    out_dir = os.path.join(os.path.dirname(__file__), "nifty_next50_basket")
    os.makedirs(out_dir, exist_ok=True)
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "trades_df"}
        for r in rows if r.get("trades", 0) > 0
    ])
    summary.to_csv(os.path.join(out_dir, "atr_sweep_summary.csv"), index=False)

    # Save per-mult trade logs
    for r in rows:
        if "trades_df" in r and not r["trades_df"].empty:
            fname = f"atr_sweep_{str(r['atr_mult']).replace('.', '_')}_trades.csv"
            r["trades_df"].to_csv(os.path.join(out_dir, fname), index=False)

    print(f"\nSummary → {out_dir}/atr_sweep_summary.csv")
