"""
EMA Regime Crossover -- Swing CNC, Rolling Universe variant (2026-07-31)

QUESTION BEING TESTED: Run 3 of the Swing CNC redesign (2026-07-30,
`ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py`) trades
a STATIC 15-name high-beta universe, hardcoded once from a single full-sample
volatility ranking and never refreshed. Karan asked: if a name OUTSIDE that
static list starts outperforming, would it ever get added? Answer (confirmed
by reading the deployed live script): no -- unlike the intraday sibling
(`ema_regime_crossover_signal_20260717094335.py`), which re-ranks its own
universe monthly via `refresh_universe()` (trailing-20-day ADV), the swing CNC
variant has no rotation mechanism at all. This script tests whether adding one
changes the result, holding every other rule byte-for-byte identical to Run 3.

THE ONLY VARIABLE UNDER TEST: universe membership rotates periodically instead
of staying fixed. Regime/entry/exit/stop/trail/max-hold/sizing/concentration-
cap/charges logic below is copied unchanged from Run 3 -- do not tune any of
those parameters in this file.

SPEC (documented per the task brief, not left implicit in code comments only):

1. RANKING BASIS: reuses the SAME metric already used to build the current
   static 15-name list --
   `ema_regime_crossover_high_beta_rank.py`'s annualized stdev of daily
   close-to-close log returns. No new metric invented. The only change
   needed for a live-safe rolling variant is that the ranking must be made
   CAUSAL: the original static rank used the full 5-year sample once
   (acceptable there only because the ranked subset was then backtested over
   that same historical window, never traded on live). A rolling variant
   must rank using only a TRAILING window ending strictly before each
   rebalance date, so nothing here uses future returns. Trailing window:
   252 trading days (~1 calendar year), a standard annualized-vol lookback;
   ranking requires >= MIN_LOOKBACK_DAYS (120 trading days, ~6 months) of
   trailing returns for a symbol to be eligible for a given rebalance date's
   ranking, else it is excluded from that one ranking round (not penalized
   permanently -- it becomes eligible again once it accumulates enough
   history).

   Bootstrap gap: at the very first rebalance date in the backtest window
   (2021-07, start of data), NO symbol yet has 120 trading days of trailing
   history, since the whole pool's price history starts at the same date.
   Rather than rank on a near-empty window (which would not be a meaningful
   volatility measure), the FIRST active period uses the existing static
   HIGH_BETA_15 list unchanged, and the rolling mechanism takes over from the
   first rebalance date that actually has sufficient trailing history for at
   least TOP_N names. This is a genuine, stated design choice (avoid ranking
   on noise), not a bug.

2. REFRESH CADENCE: two variants tested, capped at 2 total backtest runs per
   the task brief --
     - MONTHLY (first test; ROLL_CADENCE="M"): matches the intraday sibling's
       own refresh_universe() cadence, for consistency across the vault's two
       EMA-family variants. Candidate cadence despite this variant's much
       longer average hold (~12.4 trading days per Run 3 vs the intraday
       sibling's own same-day round-trip) because rotation only ever GATES
       NEW ENTRIES (see point 4) -- it never touches an open position -- so a
       faster refresh cadence costs nothing on the exit side and only
       improves how current the entry universe is.
     - QUARTERLY (second test if the first is marginal; ROLL_CADENCE="Q"):
       arguably the more natural cadence given the strategy's own 40-trading-
       day (~2-calendar-month) MAX_HOLD backstop -- a name could rotate out
       and back in within one monthly cycle while a position on it is still
       open, which is harmless under the rotation rule below but is churn
       for no benefit if trades typically run 8-24 trading days per Run 3's
       own per-exit-reason breakdown. Quarterly (63 trading days) sits closer
       to the strategy's own natural holding-period scale.
   Both share the identical ranking basis, lookback, and Top-15 cut --
   cadence is the only thing that differs between the two runs.

3. SOURCE POOL: the SAME ~59-60-name broader pool the original High-Beta
   Universe Test (and the intraday test #8 config) drew from --
   `ema_regime_crossover_high_beta_rank.py`'s own `NIFTY_50_FULL` (49 names)
   + `NEXT50_TOP10` (10 names) = 59 names. No new/different source pool
   invented. (Two names, ADANIENSOL and CGPOWER, are already known to fail to
   fetch from OpenAlgo /history in this vault's prior backtests -- expected
   to fail again here and excluded automatically by the fetch-skip path,
   same convention as Run 3.)

4. ROTATION / OPEN-POSITION HANDLING: a name dropping out of the active
   Top-15 at a rebalance date BLOCKS NEW ENTRIES ONLY. Any position already
   open on that name continues to its own stop/trail/reverse-cross/max-hold
   exit completely unchanged by the later rotation -- this follows the
   EXACT precedent already set by both existing EMA-family siblings in this
   vault: the intraday sibling's own docstring states "Regime flip does NOT
   force-close an open position -- it only blocks new entries ... an open
   position rides its own exit rule to completion" (regime, not universe,
   there, but the same non-interference principle), and the swing CNC
   variant's own concentration cap (Run 1-3) already uses this identical
   "block new entries, never force-close" pattern for the group cap. This
   is not a new invention for this script -- it is the same rule applied to
   a new gating dimension (universe membership) instead of regime or group
   cap.

5. CONCENTRATION CAP INTERACTION: `GROUP_OF` / `group_of()` are copied
   unchanged from Run 3. `group_of()` already defaults any symbol NOT
   explicitly listed to its own singleton group
   (`GROUP_OF.get(symbol, f"SINGLE_{symbol}")`) -- this generalizes
   automatically to any name that rotates into the active universe from the
   broader 59-name pool, Adani-group or not, with no code change and no
   per-rotation hardcoding required. If a newly-rotated-in name happens to
   already be one of the 5 explicitly-mapped Adani-group names, it is
   correctly capped alongside any other currently-open Adani-group position;
   any other newly-rotated name (the overwhelming majority of the 59-name
   pool) gets its own singleton group exactly as today. Verified structurally
   below (the function is untouched from Run 3), not re-implemented.

WHAT IS UNCHANGED FROM RUN 3 (copied byte-for-byte where it is pure logic,
not universe-related): regime (daily EMA200), entry (daily EMA9/20 bullish
cross while BULL), initial stop (1.5xATR), trail arm (1.0xATR)/trail distance
(2.2xATR), REVERSE_CROSS alternate exit, MAX_HOLD_DAYS=40, LONG_ONLY/CNC,
ALLOCATED_CAPITAL=Rs 2,50,000, MAX_CONCURRENT=6, ASSUMED_LEVERAGE=1,
MAX_PER_GROUP=2, compute_charges_cnc() (DP-charge-inclusive), same window
(2021-07-01 -> 2026-06-30), same daily-return Sharpe/drawdown convention.

Data: OpenAlgo `/history`, daily bars ("D"), NSE cash equity.
"""

import heapq
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

# ---- Which cadence to run: "M" (monthly) or "Q" (quarterly) ----
ROLL_CADENCE = os.getenv("ROLL_CADENCE", "M")
assert ROLL_CADENCE in ("M", "Q")

# ---- Regime / entry (test #8 family, ported to daily bars -- unchanged from Run 3) ----
REGIME_EMA = 200
ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14

# ---- Exit scheme (unchanged from Run 3) ----
INITIAL_STOP_ATR_MULT = 1.5
TRAIL_ARM_ATR_MULT = 1.0
TRAIL_ATR_MULT = 2.2
MAX_HOLD_DAYS = 40

LONG_ONLY = True

# ---- Sizing (unchanged from Run 3) ----
ALLOCATED_CAPITAL = 250_000
MAX_CONCURRENT = 6
ASSUMED_LEVERAGE = 1

# ---- Concentration cap (unchanged from Run 3) ----
MAX_PER_GROUP = 2

# ---- Rolling-universe ranking parameters (NEW, this script only) ----
TOP_N = 15
VOL_LOOKBACK_DAYS = 252       # trailing window for the ranking metric
MIN_LOOKBACK_DAYS = 120       # minimum trailing history to be ranked at all

# ---- Bootstrap universe: identical static list used until the rolling
# mechanism has enough trailing history to take over (see point 1 above) ----
HIGH_BETA_15 = [
    "ADANIGREEN", "ADANIENSOL", "ADANIENT", "ADANIPOWER", "ETERNAL", "LODHA",
    "TMCV", "VEDL", "ADANIPORTS", "CGPOWER", "MAXHEALTH", "CANBK",
    "SHRIRAMFIN", "TRENT", "JIOFIN",
]

# ---- Source pool: SAME 59-name pool as ema_regime_crossover_high_beta_rank.py ----
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
POOL = NIFTY_50_FULL + NEXT50_TOP10  # 59 names, duplicates none (checked)

# Issuer/sector group map -- unchanged from Run 3. Any symbol not explicitly
# listed defaults to its own singleton group via group_of() below -- this
# generalizes automatically to any name rotating in from POOL.
GROUP_OF = {
    "ADANIGREEN": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP",
    "ADANIENT": "ADANI_GROUP",
    "ADANIPOWER": "ADANI_GROUP",
    "ADANIPORTS": "ADANI_GROUP",
}
def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"ema_regime_crossover_swing_cnc_rolling_universe_{ROLL_CADENCE}")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- CNC (delivery) charges -- unchanged from Run 3 ----
BROKERAGE_FLAT = 0.0
BROKERAGE_PCT = 0.0
STT_PCT_DELIVERY = 0.001
EXCHANGE_PCT = 0.0000297
SEBI_PCT = 0.000001
STAMP_DUTY_PCT_DELIVERY = 0.00015
GST_PCT = 0.18
DP_CHARGE_PER_ISIN = 12.50

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


def compute_charges_cnc(entry_price, exit_price, qty):
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover
    brokerage = BROKERAGE_FLAT * 2
    stt = STT_PCT_DELIVERY * total_turnover
    exchange_chg = EXCHANGE_PCT * total_turnover
    sebi_chg = SEBI_PCT * total_turnover
    stamp_duty = STAMP_DUTY_PCT_DELIVERY * buy_turnover
    dp_charge = DP_CHARGE_PER_ISIN
    gst = GST_PCT * (brokerage + exchange_chg + sebi_chg + dp_charge)
    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst + dp_charge
    return {"brokerage": round(brokerage, 2), "stt": round(stt, 2), "exchange_chg": round(exchange_chg, 2),
            "sebi_chg": round(sebi_chg, 2), "stamp_duty": round(stamp_duty, 2), "dp_charge": round(dp_charge, 2),
            "gst": round(gst, 2), "total_charges": round(total, 2)}


print(f"Cadence: {ROLL_CADENCE} | Source pool: {len(POOL)} names | Top-N: {TOP_N} | "
      f"lookback={VOL_LOOKBACK_DAYS}d (min {MIN_LOOKBACK_DAYS}d) | LONG_ONLY={LONG_ONLY} | product=CNC | "
      f"capital=Rs{ALLOCATED_CAPITAL:,} | max_concurrent={MAX_CONCURRENT} | max_per_group={MAX_PER_GROUP} | "
      f"stop={INITIAL_STOP_ATR_MULT}xATR | trail_arm={TRAIL_ARM_ATR_MULT}xATR | trail={TRAIL_ATR_MULT}xATR | "
      f"max_hold={MAX_HOLD_DAYS}d")

price_data = {}
log_ret = {}
skipped = {}
all_trades = []

for symbol in POOL:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({err})")
        continue

    df["ema200"] = df["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    df["regime"] = np.where(df["close"] > df["ema200"], "BULL", "BEAR")
    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    price_data[symbol] = df
    log_ret[symbol] = df["log_ret"]
    print(f"  {symbol}: {len(df):,} daily bars ({df.index[0].date()} -> {df.index[-1].date()})")

# ---- Signal generation: candidate trades per symbol, universe-agnostic ----
# (same entry/exit logic as Run 3; universe eligibility is applied later, at
# the portfolio-simulation stage, so rotating in/out never touches an open
# position's own exit path -- see point 4 of the spec above.)
for symbol, df in price_data.items():
    df2 = df.dropna(subset=["atr", "ema200"])
    n = len(df2)
    for i in range(n - 1):
        row = df2.iloc[i]
        if not (row["bull_cross"] and row["regime"] == "BULL"):
            continue
        entry_ts = df2.index[i + 1]
        entry_bar = df2.iloc[i + 1]
        entry_price = entry_bar["open"]
        atr_entry = row["atr"]
        initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr_entry
        arm_level = entry_price + TRAIL_ARM_ATR_MULT * atr_entry

        trail_stop = initial_stop
        armed = False
        highest_close = entry_price
        exit_px, exit_ts, reason, hold_days = None, None, None, None
        rest = df2[df2.index > entry_ts]
        for hold_idx, (ts2, bar2) in enumerate(rest.iterrows(), start=1):
            if bar2["low"] <= trail_stop:
                exit_px, exit_ts, reason, hold_days = trail_stop, ts2, ("TRAIL_STOP" if armed else "STOP"), hold_idx
                break
            if bar2["bear_cross"]:
                exit_px, exit_ts, reason, hold_days = bar2["close"], ts2, "REVERSE_CROSS", hold_idx
                break
            if hold_idx >= MAX_HOLD_DAYS:
                exit_px, exit_ts, reason, hold_days = bar2["close"], ts2, "MAX_HOLD", hold_idx
                break
            highest_close = max(highest_close, bar2["close"])
            if not armed and bar2["close"] >= arm_level:
                armed = True
            if armed:
                candidate_trail = highest_close - TRAIL_ATR_MULT * bar2["atr"]
                trail_stop = max(trail_stop, candidate_trail)
        if exit_px is None:
            continue

        pnl_pct = (exit_px - entry_price) / entry_price
        all_trades.append({
            "symbol": symbol, "direction": "LONG", "entry_time": entry_ts,
            "entry_price": entry_price, "exit_time": exit_ts, "exit_price": exit_px,
            "pnl_pct": pnl_pct, "reason": reason, "hold_days": hold_days,
        })

if not all_trades:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
print(f"\nSignal candidates (pre-universe/portfolio-constraints, full {len(price_data)}-name pool): {len(trades_df)}")

# ---- Build rebalance calendar (causal, no lookahead) ----
master_cal = sorted(set().union(*[set(df.index) for df in price_data.values()]))
master_cal = pd.DatetimeIndex(master_cal)
cal_df = pd.DataFrame({"date": master_cal})
if ROLL_CADENCE == "M":
    cal_df["period"] = cal_df["date"].dt.to_period("M")
else:
    cal_df["period"] = cal_df["date"].dt.to_period("Q")
rebalance_dates = cal_df.groupby("period")["date"].min().sort_values().tolist()

# Combined log-return frame for vectorized trailing-vol lookups
ret_frame = pd.DataFrame(log_ret).sort_index()

periods = []  # list of (start_date, end_date_exclusive, active_set, source)
bootstrap_done = False
prev_active = None
rotation_events = []  # (rebalance_date, entered, left)

for idx, rdate in enumerate(rebalance_dates):
    end_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else master_cal[-1] + pd.Timedelta(days=1)
    trailing = ret_frame[ret_frame.index < rdate].tail(VOL_LOOKBACK_DAYS)
    counts = trailing.count()
    eligible = counts[counts >= MIN_LOOKBACK_DAYS].index.tolist()

    if len(eligible) < TOP_N:
        # Not enough trailing history yet anywhere in the pool -- bootstrap
        # with the existing static list rather than rank on a near-empty window.
        active = set(HIGH_BETA_15)
        source = "bootstrap-static-15"
    else:
        ann_vol = trailing[eligible].std() * np.sqrt(252)
        top = ann_vol.sort_values(ascending=False).head(TOP_N).index.tolist()
        active = set(top)
        source = "rolling-rank"
        bootstrap_done = True

    periods.append((rdate, end_date, active, source))
    if prev_active is not None:
        entered = active - prev_active
        left = prev_active - active
        if entered or left:
            rotation_events.append({"rebalance_date": rdate, "entered": sorted(entered), "left": sorted(left)})
    prev_active = active

print(f"\nRebalance dates ({ROLL_CADENCE}): {len(periods)} | "
      f"bootstrap periods (static-15, insufficient trailing history): "
      f"{sum(1 for p in periods if p[3] == 'bootstrap-static-15')} | "
      f"rolling-ranked periods: {sum(1 for p in periods if p[3] == 'rolling-rank')}")
print(f"Rotation events (period-to-period membership changes): {len(rotation_events)}")


def active_universe_for(ts):
    """Return the active-universe set covering entry timestamp ts."""
    for start, end, active, _source in periods:
        if start <= ts < end:
            return active
    return periods[-1][2] if periods else set()


# ---- Portfolio simulation: chronological event sim, same convention as Run 3,
# with an added universe-membership gate on NEW entries only. ----
capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT
buying_power = capital_per_trade * ASSUMED_LEVERAGE

pending_exits = []
seq = 0
open_positions = {}
group_open_count = {}

accepted = []
rejected_cap, rejected_group, rejected_universe = 0, 0, 0

for _, cand in trades_df.iterrows():
    qty = int(buying_power // cand["entry_price"])
    if qty <= 0:
        continue
    pnl_rupees = qty * cand["entry_price"] * cand["pnl_pct"]

    while pending_exits and pending_exits[0][0] <= cand["entry_time"]:
        _, ex_seq, ex_group = heapq.heappop(pending_exits)
        group_open_count[ex_group] = group_open_count.get(ex_group, 1) - 1
        del open_positions[ex_seq]

    # Universe gate -- NEW entries only. An already-open position (tracked in
    # open_positions/pending_exits, computed above independent of this gate)
    # is never touched here -- it rides its own precomputed exit regardless
    # of later rotation, per point 4 of the spec.
    active = active_universe_for(cand["entry_time"])
    if cand["symbol"] not in active:
        rejected_universe += 1
        continue

    grp = group_of(cand["symbol"])
    if len(open_positions) >= MAX_CONCURRENT:
        rejected_cap += 1
        continue
    if group_open_count.get(grp, 0) >= MAX_PER_GROUP:
        rejected_group += 1
        continue

    t = cand.to_dict()
    t["qty"] = qty
    t["pnl_rupees"] = round(pnl_rupees, 2)
    charges = compute_charges_cnc(cand["entry_price"], cand["exit_price"], qty)
    t.update(charges)
    t["net_pnl_rupees"] = round(pnl_rupees - charges["total_charges"], 2)
    t["group"] = grp
    accepted.append(t)

    seq += 1
    open_positions[seq] = {"symbol": cand["symbol"], "group": grp}
    group_open_count[grp] = group_open_count.get(grp, 0) + 1
    heapq.heappush(pending_exits, (cand["exit_time"], seq, grp))

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, f"ema_regime_crossover_swing_cnc_rolling_universe_{ROLL_CADENCE}_trades.csv"), index=False)

rotation_log = pd.DataFrame(rotation_events)
if not rotation_log.empty:
    rotation_log.to_csv(os.path.join(OUT_DIR, f"rotation_events_{ROLL_CADENCE}.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | concentration cap ({MAX_PER_GROUP}/group): {rejected_group} | "
      f"universe (not in active Top-{TOP_N}): {rejected_universe}")

# ---- Names churned most often ----
if rotation_events:
    entered_counts = pd.Series([s for ev in rotation_events for s in ev["entered"]]).value_counts()
    left_counts = pd.Series([s for ev in rotation_events for s in ev["left"]]).value_counts()
    print(f"\nTop symbols by # times ENTERED the active Top-{TOP_N}:\n{entered_counts.head(10).to_string()}")
    print(f"\nTop symbols by # times LEFT the active Top-{TOP_N}:\n{left_counts.head(10).to_string()}")

if result_df.empty:
    print("\nNo trades survived portfolio constraints.")
else:
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

    print(f"\n=== EMA Regime Crossover -- Swing CNC, Rolling Universe ({ROLL_CADENCE}) "
          f"(Rs{ALLOCATED_CAPITAL:,}, max {MAX_CONCURRENT} concurrent, max {MAX_PER_GROUP}/group) ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Avg holding period: {result_df['hold_days'].mean():.1f} trading days "
          f"(median {result_df['hold_days'].median():.0f})")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"By group: {result_df['group'].value_counts().to_dict()}")
    print(f"By symbol (top 15): {result_df['symbol'].value_counts().head(15).to_dict()}")

    print("\nBy exit reason (net of charges):")
    for reason, g in result_df.groupby("reason"):
        print(f"  {reason:14s} n={len(g):5d} ({len(g)/n*100:4.1f}%)  WR={(g.net_pnl_rupees>0).mean()*100:5.1f}%  "
              f"avg_pnl_pct={g.pnl_pct.mean()*100:+.3f}%  avg_hold={g.hold_days.mean():.1f}d  "
              f"total_net=Rs{g.net_pnl_rupees.sum():+10,.0f}")

    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    print(f"\nSharpe (gross): {sharpe(daily_gross):.2f}")
    print(f"Sharpe (net)  : {sharpe(daily_net):.2f}   (Run 3 static-universe baseline: 2.76)")

    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"\nMax drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")

    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_swing_cnc_rolling_universe_{ROLL_CADENCE}_trades.csv")
