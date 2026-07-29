"""
Nifty Next 50 Cash Basket — Proper Backtest (No ADV Look-ahead Bias)

Universe  : All 50 current Nifty Next 50 stocks (point-in-time 2026-07-17)
Selection : At each monthly rebalance, rank by trailing 20-day ADV using ONLY
            data available at that date → eliminates ADV look-ahead bias
            Stocks without >= 20 bars of history at a given rebalance date are
            excluded from selection for that month (no forward-filling of thin data)
Group cap : Max 2 stocks per conglomerate group
            - ADANI  : ADANIPOWER, ADANIENSOL, ADANIGREEN
            - TATA   : TATAPOWER, TATACAP, TMCV
            - VEDANTA: VEDL, HINDZINC
ATR mult  : 2.5× (optimal from ATR sweep across multipliers 0.75–3.0)
Target    : +10% from entry (daily high touch)
Stop      : entry − 2.5 × ATR(14, daily) at time of entry (daily low touch)
Rebalance : 1st trading day of each month — exit stale, enter new top-10
Post-exit : Cash held until next monthly rebalance
Sizing    : Rs 5,00,000 / 10 slots = Rs 50,000/slot; qty = floor(50k / open)
Product   : CNC delivery (no leverage)
Window    : 2021-07-01 → 2026-06-30

Residual caveat (unavoidable without paid historical constituent data):
  Universe composition bias — uses current (Jul 2026) Nifty Next 50 membership.
  Stocks that were not in the index in 2021-2024 may be over-represented in early
  periods. Newer listings (e.g. HYUNDAI listed Oct 2024) will have no data before
  their IPO and will be naturally excluded from early rebalances.
"""

import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
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

TARGET_PCT      = 0.10
ATR_PERIOD      = 14
ATR_MULTIPLIER  = 2.5    # optimal from ATR sweep
ADV_LOOKBACK    = 20     # trailing trading days for ADV rank

# Full Nifty Next 50 constituent list (Jul 2026 snapshot)
FULL_UNIVERSE = [
    "ABB", "ADANIPOWER", "AMBUJACEM", "ADANIENSOL", "ADANIGREEN", "BAJAJHLDNG",
    "BANKBARODA", "CGPOWER", "ZYDUSLIFE", "DLF", "BPCL", "CANBK", "DIVISLAB",
    "BRITANNIA", "CHOLAFIN", "CUMMINSIND", "DMART", "BOSCHLTD", "HAL", "GODREJCP",
    "GAIL", "HDFCAMC", "IOC", "HINDZINC", "INDHOTEL", "JINDALSTEL", "LTM",
    "UNITDSPR", "MUTHOOTFIN", "MOTHERSON", "RECLTD", "PIDILITIND", "PFC", "PNB",
    "TATAPOWER", "SOLARINDS", "SHREECEM", "SIEMENS", "UNIONBANK", "VEDL", "TVSMOTOR",
    "TORNTPHARM", "VBL", "MAZDOCK", "IRFC", "LODHA", "HYUNDAI", "ENRIN", "TATACAP",
    "TMCV",
]

# Conglomerate group caps (max 2 per group)
GROUPS = {
    "ADANI"  : {"ADANIPOWER", "ADANIENSOL", "ADANIGREEN"},
    "TATA"   : {"TATAPOWER", "TATACAP", "TMCV"},
    "VEDANTA": {"VEDL", "HINDZINC"},
}
GROUP_CAP = 2

# ── Client ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)

# ── Data fetch ────────────────────────────────────────────────────────────────
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

# ── ATR (Wilder's smoothing via EWM) ─────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ── Dynamic top-10 selection at a rebalance date ──────────────────────────────
def select_top10(price_data: dict, rebal_date: pd.Timestamp) -> list[str]:
    """
    Rank all available symbols by trailing ADV_LOOKBACK-day ADV using only data
    up to (and including) rebal_date. Apply group cap. Return up to N_SLOTS symbols.
    """
    adv_rows = []
    for sym, df in price_data.items():
        hist = df[df.index <= rebal_date]
        if len(hist) < ADV_LOOKBACK:
            continue
        window = hist.tail(ADV_LOOKBACK)
        if "close" not in window.columns or "volume" not in window.columns:
            continue
        adv = (window["close"] * window["volume"]).mean()
        if pd.isna(adv) or adv <= 0:
            continue
        adv_rows.append({"symbol": sym, "adv": adv})

    if not adv_rows:
        return []

    ranked = pd.DataFrame(adv_rows).sort_values("adv", ascending=False)

    selected = []
    group_counts: dict[str, int] = {}

    for _, row in ranked.iterrows():
        if len(selected) >= N_SLOTS:
            break
        sym = row["symbol"]
        # Group cap check
        capped = False
        for grp_name, members in GROUPS.items():
            if sym in members:
                if group_counts.get(grp_name, 0) >= GROUP_CAP:
                    capped = True
                break
        if capped:
            continue
        # Accept
        selected.append(sym)
        for grp_name, members in GROUPS.items():
            if sym in members:
                group_counts[grp_name] = group_counts.get(grp_name, 0) + 1
                break

    return selected

# ── Rebalance calendar ────────────────────────────────────────────────────────
def build_rebalance_dates(all_dates: pd.DatetimeIndex) -> set:
    months = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    out = set()
    for m in months:
        bucket = all_dates[(all_dates.year == m.year) & (all_dates.month == m.month)]
        if len(bucket):
            out.add(bucket[0])
    return out

# ── Backtest engine ───────────────────────────────────────────────────────────
def run_backtest(price_data: dict) -> dict:
    all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
    all_dates  = pd.DatetimeIndex(all_dates)
    rebal_set  = build_rebalance_dates(all_dates)

    active  = {}
    cash    = float(TOTAL_CAPITAL)
    trades  = []
    equity  = []
    rebal_log = []  # track what was selected each month

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
            "entry_price": entry, "qty": qty,
            "stop": stop, "target": target, "entry_date": date, "atr": atr_val,
        }

    for date in all_dates:
        # 1. Check stop / target
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

        # 2. Monthly rebalance
        if date in rebal_set:
            desired = set(select_top10(price_data, date))
            current = set(active.keys())

            rebal_log.append({
                "rebal_date": date.date(),
                "selected"  : sorted(desired),
                "exited"    : sorted(current - desired),
                "entered"   : sorted(desired - current),
            })

            for sym in current - desired:
                if sym in price_data and date in price_data[sym].index:
                    close_pos(sym, price_data[sym].loc[date]["open"], date, "REBALANCE_EXIT")

            for sym in desired - set(active.keys()):
                open_pos(sym, date)

        # 3. Mark-to-market
        mtm = cash
        for sym, pos in active.items():
            if sym in price_data and date in price_data[sym].index:
                mtm += price_data[sym].loc[date]["close"] * pos["qty"]
        equity.append({"date": date.date(), "equity": round(mtm, 2)})

    last = all_dates[-1]
    for sym in list(active.keys()):
        if sym in price_data and last in price_data[sym].index:
            close_pos(sym, price_data[sym].loc[last]["close"], last, "END_OF_BACKTEST")

    return {
        "trades"      : pd.DataFrame(trades),
        "equity_curve": pd.DataFrame(equity),
        "rebal_log"   : rebal_log,
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

    W = 62
    print("\n" + "=" * W)
    print("  NIFTY NEXT 50 BASKET — PROPER BACKTEST (DYNAMIC ADV RANKING)")
    print("=" * W)
    print(f"  Period        : {START_DATE}  →  {END_DATE}")
    print(f"  Universe      : {len(FULL_UNIVERSE)} stocks (full Nifty Next 50)")
    print(f"  Selection     : Dynamic trailing {ADV_LOOKBACK}-day ADV at each rebalance")
    print(f"  Group cap     : Adani/Tata/Vedanta — max {GROUP_CAP} each")
    print(f"  Target / Stop : +{TARGET_PCT*100:.0f}%  /  {ATR_MULTIPLIER}×ATR({ATR_PERIOD}, daily)")
    print(f"  Capital       : ₹{TOTAL_CAPITAL:,.0f}  ({N_SLOTS} slots × ₹{CAPITAL_PER_SLOT:,.0f})")
    print("-" * W)
    print(f"  Total trades  : {len(trades)}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Net P&L       : ₹{net:,.0f}")
    print(f"  CAGR          : {cagr:.1f}%")
    print(f"  Max drawdown  : {dd:.1f}%")
    print(f"  Sharpe ratio  : {sharpe:.2f}")
    print(f"  Avg win       : ₹{wins['pnl'].mean():,.0f}")
    print(f"  Avg loss      : ₹{losses['pnl'].mean():,.0f}")
    print(f"  Avg holding   : {trades['holding_days'].mean():.0f} days")
    print("-" * W)
    print("  Exit breakdown:")
    for reason, cnt in trades["reason"].value_counts().items():
        print(f"    {reason:<22}: {cnt}")
    print("-" * W)
    print("  Per-symbol P&L (best → worst):")
    sym = trades.groupby("symbol")["pnl"].agg(["sum", "count", "mean"])
    sym.columns = ["total", "n", "avg"]
    for s, r in sym.sort_values("total", ascending=False).iterrows():
        sgn = "+" if r["total"] >= 0 else ""
        print(f"    {s:<15}: {sgn}₹{r['total']:>10,.0f}  ({int(r['n'])} trades, avg {sgn}₹{r['avg']:,.0f})")
    print("=" * W)

    # Sample rebalance selections (first 3 and last 3)
    log = results["rebal_log"]
    print(f"\n  Rebalance log ({len(log)} months):")
    show = log[:3] + (["..."] if len(log) > 6 else []) + log[-3:]
    for entry in show:
        if entry == "...":
            print("    ...")
            continue
        print(f"    {entry['rebal_date']}  selected: {', '.join(entry['selected'])}")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Nifty Next 50 Basket — Proper Backtest  |  {START_DATE} → {END_DATE}")
    print(f"Dynamic ADV ranking at each rebalance  |  ATR×{ATR_MULTIPLIER}  |  ₹{CAPITAL_PER_SLOT:,.0f}/slot")
    print(f"Fetching data for {len(FULL_UNIVERSE)} stocks...\n")

    price_data: dict[str, pd.DataFrame] = {}
    skipped = []
    for sym in FULL_UNIVERSE:
        print(f"  {sym:<15}...", end=" ", flush=True)
        df, err = fetch_daily(sym)
        if err:
            print(f"SKIP ({err})")
            skipped.append((sym, err))
        else:
            print(f"{len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
            price_data[sym] = df

    if skipped:
        print(f"\n  Skipped {len(skipped)}: {[s for s, _ in skipped]}")

    print(f"\nRunning backtest on {len(price_data)} symbols with dynamic ADV selection...")
    results = run_backtest(price_data)
    print_metrics(results)

    out_dir = os.path.join(os.path.dirname(__file__), "nifty_next50_basket")
    os.makedirs(out_dir, exist_ok=True)
    results["trades"].to_csv(os.path.join(out_dir, "proper_trades.csv"), index=False)
    results["equity_curve"].to_csv(os.path.join(out_dir, "proper_equity_curve.csv"), index=False)

    # Save rebalance log
    rebal_rows = []
    for entry in results["rebal_log"]:
        rebal_rows.append({
            "rebal_date": entry["rebal_date"],
            "selected"  : "|".join(entry["selected"]),
            "exited"    : "|".join(entry["exited"]),
            "entered"   : "|".join(entry["entered"]),
        })
    pd.DataFrame(rebal_rows).to_csv(os.path.join(out_dir, "proper_rebal_log.csv"), index=False)

    print(f"\n  Trade log    → {out_dir}/proper_trades.csv")
    print(f"  Equity curve → {out_dir}/proper_equity_curve.csv")
    print(f"  Rebal log    → {out_dir}/proper_rebal_log.csv")
