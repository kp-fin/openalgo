"""
VP Value Area Scalp -- Bank Nifty 2m Volume Profile scalp, first backtest
(2026-08-02)

SOURCE: strategy idea captured from a Dhan YouTube video ("Bank Nifty
Traders Keep Ignoring This Setup | High-Probability Trading Strategy",
2026-07-31, https://www.youtube.com/watch?v=BPDusRqdj8A). Spec locked with
Karan 2026-08-02 -- see indices-system/strategies/vp_value_area_scalp.md.

INSTRUMENT / DATA: BANKNIFTY futures (current front-month contract symbol,
BANKNIFTY25AUG26FUT on NFO) used as a spot-points proxy -- chosen over the
BANKNIFTY index because the index has zero traded volume (can't build a
Volume Profile from it) and no continuous expired-futures history is
queryable via the broker's instrument master. 1-minute bars are available
for this contract from ~2026-06-01 to ~2026-07-31 (its live trading life
so far) -- roughly 40 sessions. This is a SMALL SAMPLE, explicitly accepted
by Karan over a "current-month-only" alternative that would have been even
smaller -- read every stat below as inconclusive-eligible, not a verdict.

RULE SET (locked 2026-08-02, all defaults -- nothing swept in this first run):
  - Base bars: 1m, resampled to 2m in-script.
  - Volume Profile: Fixed Range over 09:15-09:30 (first 15m -- 7-8 raw 1m
    bars depending on session start alignment), computed from real futures
    volume. VAH/VAL = the top/bottom of the value area (70% of volume,
    standard definition), NOT simply high/low of the range.
  - Trend filter: 10 EMA vs 20 EMA on the 2m chart at signal time.
    10>20 -> bullish bias only. 10<20 -> bearish bias only.
  - Entry window: 09:30-10:45 IST. One trade per day (first valid signal).
  - Long entry: 2m close back inside range after a bullish bias pullback
    touches/pierces VAL, AND the pullback's last two 2m candles form a
    bullish engulfing pattern (candle[-1] bearish body, candle[0] bullish
    body whose open <= candle[-1] close and close >= candle[-1] open).
  - Short entry: mirror at VAH with a bearish engulfing pattern, bearish
    bias only.
  - Stop: at the far side of the value area boundary the entry was built
    from (VAL - buffer for longs, VAH + buffer for shorts). Buffer = 0
    (exact boundary) per the video's literal "based on the volume area
    boundaries" language -- no extra padding invented.
  - Partial exit: book 50% of size at the OPPOSITE zone (VAH for longs,
    VAL for shorts).
  - Trail (remaining 50%): stop moves to breakeven (entry price) the
    instant the partial fills, then trails behind the low (longs) / high
    (shorts) of the last 3 closed 2m candles (simple structure trail) --
    Karan's chosen default over "no trail, hold to session end".
  - Hard exit: 15:15 IST for any still-open position (matches ORB_Spread
    convention, not separately specified in the video).
  - No signal in the 09:30-10:45 window -> no trade that day (video's own
    discipline rule, naturally falls out of the entry-window constraint).

OUTPUT: per-trade CSV + console summary stats to
indices-system/logs/verification/ (this strategy's home system is
indices-system/, per its registry entry).
"""

import os

import numpy as np
import pandas as pd

# ---- config ----
SYMBOL = "BANKNIFTY25AUG26FUT"
EXCHANGE = "NFO"
START_DATE = "2026-06-01"
END_DATE = "2026-07-31"

SESSION_START = "09:15"
PROFILE_END = "09:30"
ENTRY_WINDOW_END = "10:45"
HARD_EXIT = "15:15"

EMA_FAST = 10
EMA_SLOW = 20
VALUE_AREA_PCT = 0.70
TRAIL_LOOKBACK = 3  # closed 2m candles

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "indices-system", "logs", "verification",
)
OUT_DIR = os.path.normpath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

API_KEY = os.getenv("OPENALGO_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENALGO_API_KEY environment variable before running.")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

from openalgo import api as openalgo_api

client = openalgo_api(api_key=API_KEY, host=HOST)


def fetch_1m():
    resp = client.history(
        symbol=SYMBOL, exchange=EXCHANGE, interval="1m",
        start_date=START_DATE, end_date=END_DATE,
    )
    if isinstance(resp, dict):
        if resp.get("status") != "success":
            raise SystemExit(f"api error: {resp.get('message', resp)}")
        df = pd.DataFrame(resp.get("data", []))
        if df.empty:
            raise SystemExit("no data returned")
        ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
        df["datetime"] = pd.to_datetime(df[ts_col])
        df = df.set_index("datetime")
    else:
        df = resp
        if df is None or df.empty:
            raise SystemExit("no data returned")
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def resample_2m(df_1m):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df_1m.resample("2min", label="left", closed="left").agg(agg).dropna(subset=["open"])
    return out


def value_area(profile_1m_day):
    """Fixed Range Volume Profile on 09:15-09:30 1m bars for one day.
    Returns (vah, val) using a simple price-bin volume distribution."""
    win = profile_1m_day.between_time(SESSION_START, PROFILE_END, inclusive="left")
    if win.empty or win["volume"].sum() <= 0:
        return None, None
    lo, hi = win["low"].min(), win["high"].max()
    if hi <= lo:
        return None, None
    n_bins = 30
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_vol = np.zeros(n_bins)
    for _, row in win.iterrows():
        # distribute each bar's volume evenly across the bins it spans
        b_lo, b_hi, vol = row["low"], row["high"], row["volume"]
        if b_hi <= b_lo or vol <= 0:
            continue
        lo_idx = np.searchsorted(bin_edges, b_lo, side="right") - 1
        hi_idx = np.searchsorted(bin_edges, b_hi, side="right") - 1
        lo_idx = max(0, min(lo_idx, n_bins - 1))
        hi_idx = max(0, min(hi_idx, n_bins - 1))
        span = hi_idx - lo_idx + 1
        for i in range(lo_idx, hi_idx + 1):
            bin_vol[i] += vol / span
    poc_idx = int(np.argmax(bin_vol))
    total_vol = bin_vol.sum()
    target = total_vol * VALUE_AREA_PCT
    lo_i, hi_i = poc_idx, poc_idx
    covered = bin_vol[poc_idx]
    while covered < target and (lo_i > 0 or hi_i < n_bins - 1):
        expand_lo = bin_vol[lo_i - 1] if lo_i > 0 else -1
        expand_hi = bin_vol[hi_i + 1] if hi_i < n_bins - 1 else -1
        if expand_hi >= expand_lo:
            hi_i += 1
            covered += bin_vol[hi_i]
        else:
            lo_i -= 1
            covered += bin_vol[lo_i]
    val = bin_edges[lo_i]
    vah = bin_edges[hi_i + 1]
    return vah, val


def is_bullish_engulfing(prev, cur):
    prev_bear = prev["close"] < prev["open"]
    cur_bull = cur["close"] > cur["open"]
    return (prev_bear and cur_bull
            and cur["open"] <= prev["close"]
            and cur["close"] >= prev["open"])


def is_bearish_engulfing(prev, cur):
    prev_bull = prev["close"] > prev["open"]
    cur_bear = cur["close"] < cur["open"]
    return (prev_bull and cur_bear
            and cur["open"] >= prev["close"]
            and cur["close"] <= prev["open"])


def run_day(day_2m, day_1m):
    vah, val = value_area(day_1m)
    if vah is None:
        return None, "no_volume_profile"

    day_2m = day_2m.copy()
    day_2m["ema_fast"] = day_2m["close"].ewm(span=EMA_FAST, adjust=False).mean()
    day_2m["ema_slow"] = day_2m["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    entry_start = day_2m.index[0].normalize() + pd.Timedelta(PROFILE_END + ":00")
    entry_end = day_2m.index[0].normalize() + pd.Timedelta(ENTRY_WINDOW_END + ":00")
    hard_exit_ts = day_2m.index[0].normalize() + pd.Timedelta(HARD_EXIT + ":00")

    window = day_2m[(day_2m.index >= entry_start) & (day_2m.index < entry_end)]
    if len(window) < 2:
        return None, "no_entry_window_data"

    for i in range(1, len(window)):
        cur = window.iloc[i]
        prev = window.iloc[i - 1]
        ts = window.index[i]
        bullish_bias = cur["ema_fast"] > cur["ema_slow"]
        bearish_bias = cur["ema_fast"] < cur["ema_slow"]

        if (bullish_bias and prev["low"] <= val and cur["close"] > val
                and is_bullish_engulfing(prev, cur)):
            entry_price = cur["close"]
            stop = val
            partial_target = vah
            direction = "LONG"
        elif (bearish_bias and prev["high"] >= vah and cur["close"] < vah
                and is_bearish_engulfing(prev, cur)):
            entry_price = cur["close"]
            stop = vah
            partial_target = val
            direction = "SHORT"
        else:
            continue

        risk_pts = abs(entry_price - stop)
        if risk_pts <= 0:
            continue

        trade = simulate_trade(day_2m, ts, direction, entry_price, stop,
                                partial_target, hard_exit_ts)
        return trade, "trade"

    return None, "no_signal"


def simulate_trade(day_2m, entry_ts, direction, entry_price, stop, partial_target, hard_exit_ts):
    path = day_2m[day_2m.index > entry_ts]
    path = path[path.index <= hard_exit_ts]

    remaining_qty = 1.0
    filled_partial = False
    trail_stop = stop
    pnl_pts = 0.0
    exit_reason = None
    exit_ts = None
    exit_price = None
    closes = []

    for ts, bar in path.iterrows():
        closes.append(bar)
        if direction == "LONG":
            if not filled_partial:
                if bar["low"] <= trail_stop:
                    exit_price = trail_stop
                    pnl_pts = exit_price - entry_price
                    exit_reason = "stop"
                    exit_ts = ts
                    break
                if bar["high"] >= partial_target:
                    pnl_pts += 0.5 * (partial_target - entry_price)
                    remaining_qty = 0.5
                    filled_partial = True
                    trail_stop = entry_price
            else:
                if len(closes) >= TRAIL_LOOKBACK + 1:
                    recent = day_2m[(day_2m.index > entry_ts) & (day_2m.index <= ts)].tail(TRAIL_LOOKBACK + 1).iloc[:-1]
                    if not recent.empty:
                        trail_stop = max(trail_stop, recent["low"].min())
                if bar["low"] <= trail_stop:
                    exit_price = trail_stop
                    pnl_pts += remaining_qty * (exit_price - entry_price)
                    exit_reason = "trail_stop"
                    exit_ts = ts
                    break
        else:  # SHORT
            if not filled_partial:
                if bar["high"] >= trail_stop:
                    exit_price = trail_stop
                    pnl_pts = entry_price - exit_price
                    exit_reason = "stop"
                    exit_ts = ts
                    break
                if bar["low"] <= partial_target:
                    pnl_pts += 0.5 * (entry_price - partial_target)
                    remaining_qty = 0.5
                    filled_partial = True
                    trail_stop = entry_price
            else:
                if len(closes) >= TRAIL_LOOKBACK + 1:
                    recent = day_2m[(day_2m.index > entry_ts) & (day_2m.index <= ts)].tail(TRAIL_LOOKBACK + 1).iloc[:-1]
                    if not recent.empty:
                        trail_stop = min(trail_stop, recent["high"].max())
                if bar["high"] >= trail_stop:
                    exit_price = trail_stop
                    pnl_pts += remaining_qty * (entry_price - exit_price)
                    exit_reason = "trail_stop"
                    exit_ts = ts
                    break

    if exit_reason is None:
        # ran to hard exit without being stopped/trailed out
        last_bar = path.iloc[-1] if not path.empty else None
        if last_bar is not None:
            exit_price = last_bar["close"]
            exit_ts = path.index[-1]
            if direction == "LONG":
                leg_pnl = exit_price - entry_price
            else:
                leg_pnl = entry_price - exit_price
            pnl_pts += remaining_qty * leg_pnl
            exit_reason = "hard_exit"
        else:
            exit_price = entry_price
            exit_ts = entry_ts
            exit_reason = "no_bars_after_entry"

    return {
        "entry_time": entry_ts,
        "direction": direction,
        "entry_price": entry_price,
        "stop": stop,
        "partial_target": partial_target,
        "partial_filled": filled_partial,
        "exit_time": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_pts": pnl_pts,
    }


def main():
    print(f"Fetching 1m data for {SYMBOL} ({EXCHANGE}) {START_DATE} -> {END_DATE} ...")
    df_1m = fetch_1m()
    print(f"  {len(df_1m)} 1m bars fetched")

    df_2m = resample_2m(df_1m)
    print(f"  {len(df_2m)} 2m bars after resample")

    trades = []
    skip_reasons = {}
    for day, day_1m in df_1m.groupby(df_1m.index.date):
        day_2m = df_2m[df_2m.index.date == day]
        if day_2m.empty:
            continue
        trade, reason = run_day(day_2m, day_1m)
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if trade is not None:
            trade["date"] = day
            trades.append(trade)

    print("\nDay outcomes:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    if not trades:
        print("\nNo trades generated. Nothing to summarize.")
        return

    df_trades = pd.DataFrame(trades)
    csv_path = os.path.join(OUT_DIR, "vp_value_area_scalp_trades.csv")
    df_trades.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_trades)} trades -> {csv_path}")

    n = len(df_trades)
    wins = (df_trades["pnl_pts"] > 0).sum()
    wr = wins / n * 100
    gp = df_trades.loc[df_trades["pnl_pts"] > 0, "pnl_pts"].sum()
    gl = abs(df_trades.loc[df_trades["pnl_pts"] <= 0, "pnl_pts"].sum())
    pf = min(gp / gl, 99.9) if gl > 0 else 99.9
    avg = df_trades["pnl_pts"].mean()

    print("\n== VP Value Area Scalp -- backtest summary ==")
    print(f"  Symbol / window : {SYMBOL} {START_DATE} -> {END_DATE}")
    print(f"  Trades          : {n}")
    print(f"  Win rate        : {wr:.1f}%")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Avg P&L (pts)   : {avg:+.2f}")
    print(f"  Total P&L (pts) : {df_trades['pnl_pts'].sum():+.2f}")
    print(f"  By direction:\n{df_trades.groupby('direction')['pnl_pts'].agg(['count', 'mean', 'sum'])}")
    print(f"  By exit reason:\n{df_trades['exit_reason'].value_counts()}")

    md_path = os.path.join(OUT_DIR, "vp_value_area_scalp_backtest_2026-08-02.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VP Value Area Scalp -- Backtest (2026-08-02)\n\n")
        f.write(f"Symbol: {SYMBOL} ({EXCHANGE}), spot-points proxy\n\n")
        f.write(f"Window: {START_DATE} -> {END_DATE} (current front-month futures contract's "
                "live 1m history -- ~40 sessions, small sample, accepted tradeoff for real "
                "traded volume over a longer synthetic series)\n\n")
        f.write(f"- Trades: {n}\n")
        f.write(f"- Win rate: {wr:.1f}%\n")
        f.write(f"- Profit factor: {pf:.2f}\n")
        f.write(f"- Avg P&L: {avg:+.2f} pts\n")
        f.write(f"- Total P&L: {df_trades['pnl_pts'].sum():+.2f} pts\n\n")
        f.write("Day outcomes:\n\n")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            f.write(f"- {reason}: {count}\n")
        f.write("\nSee `vp_value_area_scalp_trades.csv` in this folder for the full trade log.\n")
    print(f"Saved summary -> {md_path}")


if __name__ == "__main__":
    main()
