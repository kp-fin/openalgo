"""
EMA Regime Crossover — SWING, LONG-ONLY — stop-width sweep at fixed target=8xATR

RS_MOMENTUM_SLOPE_LOOKBACK locked at 2 (see the lookback sweep). Target fixed
at 8xATR (still the best edge-of-grid point from the ATR sweep --
ema_regime_crossover_swing_long_only_atr_sweep.py -- which hadn't found a
true peak yet). This script isolates the STOP axis instead, since the
earlier full grid showed win rate is driven by stop width, not target width:
at fixed target, widening the stop raises WR but lowers PF. Karan's request
2026-07-19, to see how far WR can climb before PF collapses.
"""

import os
import warnings

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

ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14
TOTAL_CAPITAL = 10_000_000
POSITION_PCT = 0.125
MAX_CONCURRENT = 6

RS_ZSCORE_WINDOW = 10
RS_SMOOTH_WINDOW = 2
RS_MOMENTUM_SLOPE_LOOKBACK = 2

TARGET_MULT = 8.0
STOP_MULTS = [1.0, 1.36, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

NIFTY50_TOP10 = ["HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
                 "TCS", "SBIN", "ETERNAL", "BAJFINANCE", "LT"]
NEXT50_TOP10 = ["ADANIPOWER", "VEDL", "CGPOWER", "LODHA", "ADANIENSOL",
                "ADANIGREEN", "TORNTPHARM", "TVSMOTOR", "CANBK", "TMCV"]
UNIVERSE = NIFTY50_TOP10 + NEXT50_TOP10
INDEX_FOR = {**{s: "NIFTY" for s in NIFTY50_TOP10}, **{s: "NIFTYNXT50" for s in NEXT50_TOP10}}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_regime_crossover_swing_long_only_stop_sweep_target8")
os.makedirs(OUT_DIR, exist_ok=True)

from openalgo import api as openalgo_api
client = openalgo_api(api_key=API_KEY, host=HOST)


def _fetch_daily(symbol, exchange):
    try:
        resp = client.history(symbol=symbol, exchange=exchange, interval="D",
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
    df.index = df.index.normalize()
    return df.sort_index(), None


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_rs_ratio_momentum(stock_close, index_close):
    rs = 100 * (stock_close / index_close)
    rs_ratio_raw = 100 + (rs - rs.rolling(RS_ZSCORE_WINDOW).mean()) / rs.rolling(RS_ZSCORE_WINDOW).std()
    rs_ratio = rs_ratio_raw.rolling(RS_SMOOTH_WINDOW).mean()
    mom_raw = 100 + (rs_ratio - rs_ratio.rolling(RS_ZSCORE_WINDOW).mean()) / rs_ratio.rolling(RS_ZSCORE_WINDOW).std()
    rs_momentum = mom_raw.rolling(RS_SMOOTH_WINDOW).mean()
    return rs_ratio, rs_momentum


index_data = {}
for idx_symbol in ("NIFTY", "NIFTYNXT50"):
    df_idx, err = _fetch_daily(idx_symbol, "NSE_INDEX")
    if err:
        raise SystemExit(f"Could not fetch index {idx_symbol}: {err}")
    index_data[idx_symbol] = df_idx

print(f"Universe: {len(UNIVERSE)} names (LONG-ONLY, LOOKBACK=2, TARGET=8.0xATR fixed, sweeping STOP)")
symbol_data = {}
skipped = {}
for symbol in UNIVERSE:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    idx_df = index_data[INDEX_FOR[symbol]]
    combined_close = pd.DataFrame({"stock": df["close"], "index": idx_df["close"]}).dropna()
    rs_ratio, rs_momentum = compute_rs_ratio_momentum(combined_close["stock"], combined_close["index"])

    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])
    df["rs_ratio"] = rs_ratio.reindex(df.index)
    df["rs_momentum"] = rs_momentum.reindex(df.index)
    df = df.dropna(subset=["atr", "rs_ratio", "rs_momentum"])

    rs_momentum_rising = df["rs_momentum"] > df["rs_momentum"].shift(RS_MOMENTUM_SLOPE_LOOKBACK)
    bull_regime = (df["rs_ratio"] > 100) & ((df["rs_momentum"] > 100) | rs_momentum_rising)
    df["regime"] = np.where(bull_regime, "BULL", "NOT_BULL")

    symbol_data[symbol] = df
    print(f"  {symbol} (vs {INDEX_FOR[symbol]}): {len(df):,} usable daily bars")


def run_backtest(stop_mult, target_mult):
    all_trades = []
    for symbol, df in symbol_data.items():
        n = len(df)
        for i in range(n - 1):
            if not (df["bull_cross"].iloc[i] and df["regime"].iloc[i] == "BULL"):
                continue
            entry_ts = df.index[i + 1]
            entry_price = df.iloc[i + 1]["open"]
            atr = df["atr"].iloc[i]
            stop_px = entry_price - stop_mult * atr
            target_px = entry_price + target_mult * atr

            exit_px, exit_ts, reason = None, None, None
            rest = df[df.index > entry_ts]
            for ts2, bar2 in rest.iterrows():
                if bar2["low"] <= stop_px:
                    exit_px, exit_ts, reason = stop_px, ts2, "STOP"
                    break
                if bar2["high"] >= target_px:
                    exit_px, exit_ts, reason = target_px, ts2, "TARGET"
                    break
                if bar2["bear_cross"]:
                    exit_px, exit_ts, reason = bar2["close"], ts2, "REVERSE_CROSS"
                    break
            if exit_px is None:
                if len(rest) == 0:
                    continue
                last_ts, last_bar = rest.index[-1], rest.iloc[-1]
                exit_px, exit_ts, reason = last_bar["close"], last_ts, "END_OF_DATA"

            pnl_pct = (exit_px - entry_price) / entry_price
            all_trades.append({
                "symbol": symbol, "entry_time": entry_ts, "entry_price": entry_price,
                "exit_time": exit_ts, "exit_price": exit_px,
                "pnl_pct": pnl_pct * 100, "reason": reason,
            })

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)

    position_size = TOTAL_CAPITAL * POSITION_PCT
    open_positions = []
    accepted = []
    for _, trade in trades_df.iterrows():
        open_positions = [p for p in open_positions if p > trade["entry_time"]]
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        open_positions.append(trade["exit_time"])
        qty = int(position_size // trade["entry_price"])
        pnl_rupees = qty * trade["entry_price"] * (trade["pnl_pct"] / 100)
        t = trade.to_dict()
        t["qty"] = qty
        t["pnl_rupees"] = pnl_rupees
        accepted.append(t)

    result_df = pd.DataFrame(accepted)
    if result_df.empty:
        return None

    n = len(result_df)
    wr = (result_df["pnl_pct"] > 0).sum() / n * 100
    total_rupees = result_df["pnl_rupees"].sum()
    gw = result_df[result_df["pnl_pct"] > 0]["pnl_rupees"].sum()
    gl = abs(result_df[result_df["pnl_pct"] <= 0]["pnl_rupees"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    by_symbol = result_df.groupby("symbol")["pnl_rupees"].sum()
    profitable_symbols = (by_symbol > 0).sum()
    avg_holding = (pd.to_datetime(result_df["exit_time"]) - pd.to_datetime(result_df["entry_time"])).dt.days.mean()
    reason_counts = result_df["reason"].value_counts().to_dict()
    return {
        "stop_mult": stop_mult, "target_mult": target_mult, "trades": n,
        "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
        "avg_pnl_pct": round(result_df["pnl_pct"].mean(), 3),
        "total_rupees": round(total_rupees, 0),
        "profitable_symbols": f"{profitable_symbols}/{len(by_symbol)}",
        "avg_holding_days": round(avg_holding, 1),
        "stop_count": reason_counts.get("STOP", 0),
        "target_count": reason_counts.get("TARGET", 0),
        "reverse_count": reason_counts.get("REVERSE_CROSS", 0),
    }


summary_rows = []
for stop_mult in STOP_MULTS:
    summary = run_backtest(stop_mult, TARGET_MULT)
    if summary is None:
        print(f"stop={stop_mult}: no trades")
        continue
    summary_rows.append(summary)
    print(f"stop={stop_mult:>4} | trades={summary['trades']:>4} | WR={summary['win_rate']:>5.1f}% | "
          f"PF={summary['profit_factor']:>5.2f} | avg={summary['avg_pnl_pct']:>+7.3f}% | "
          f"total=Rs {summary['total_rupees']:>+13,.0f} | profitable={summary['profitable_symbols']} | "
          f"avg_hold={summary['avg_holding_days']}d | STOP/TARGET/REVERSE={summary['stop_count']}/{summary['target_count']}/{summary['reverse_count']}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "stop_sweep_target8_summary.csv"), index=False)
print(f"\nSummary written to {os.path.join(OUT_DIR, 'stop_sweep_target8_summary.csv')}")
print(f"Skipped symbols: {skipped}")
