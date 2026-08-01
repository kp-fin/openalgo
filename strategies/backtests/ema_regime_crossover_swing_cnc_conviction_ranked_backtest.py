"""
EMA Regime Crossover -- Swing CNC, Conviction-Ranked Entry Selection Test
(2026-08-01, Karan-requested follow-on to the Middle-Ground Universe Size rejection)

CONTEXT: today's universe-sizing campaign established that widening the
candidate pool under a fixed MAX_CONCURRENT=6 slot cap dilutes trade quality
via first-come-first-served entry -- Top-25/Top-30 rejected (Sharpe(net) 1.86
and 1.38, monotonically worse than Run 3's 2.76), full-Nifty-100-unranked
rejected decisively (Sharpe(net) 0.55). The diagnostic in both rejections
showed concurrency-cap rejections outnumbering concentration-cap rejections
~12-14:1 -- MORE signals competing for the SAME 6 slots, with no mechanism to
prefer the better ones, is the dilution mechanism.

Karan's question: with the 6-slot cap held fixed, can a WIDE candidate pool
still be used productively if simultaneous signals are ranked by conviction
and only the best fill the available slots, instead of strict
first-come-first-served? THIS SCRIPT tests exactly that -- the ONLY structural
change vs. every other backtest today is the portfolio-construction step
(batch-and-rank by conviction instead of one-at-a-time FIFO). Regime/entry/exit
logic, sizing, charges, MAX_CONCURRENT=6, MAX_PER_GROUP=2, ALLOCATED_CAPITAL=
250,000, and the 2021-07-01->2026-06-30 window are held byte-for-byte identical
to Run 3 (`ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py`).

CANDIDATE POOL: full Nifty 100 (same UNIVERSE/GROUP_OF as
`ema_regime_crossover_swing_cnc_nifty100_backtest.py`, today's test #4 -- reused
verbatim, not re-derived) -- the widest pool tested today, so any benefit from
ranking has the most raw material to work with.

CONVICTION SCORE (computed at the entry SIGNAL bar, i.e. the bar whose
bull_cross+BULL regime fires the candidate -- one bar before the actual t+1
open-price entry, so this is a purely backward-looking quantity, no lookahead):

    conviction = (signal_bar.close - signal_bar.ema200) / signal_bar.atr14

Interpretation: how many ATRs above the 200-EMA regime filter the stock is
sitting at the moment its 9/20 EMA crossover fires. Both close, ema200, and
atr14 at that bar are computed from historical data up to and including that
bar only (ema/atr are causal rolling calculations, standard pandas .ewm --
no forward-looking window). The entry itself still executes at the NEXT bar's
open (t+1), same as every other script in this campaign -- conviction is
simply a ranking key attached to each candidate, not a change to the fill
price or timing.

PORTFOLIO-CONSTRUCTION CHANGE (the actual experiment): every prior script in
this campaign (including Run 3 and today's other 3 tests) processes candidates
as a single chronologically-sorted event stream, one at a time, via a
heapq-based `pending_exits` mechanism -- first-come-first-served entry, no
visibility into what else fires that same day. That structure CANNOT express
"rank against today's other candidates," because ranking requires seeing all
of a day's candidates before deciding which to take. This script therefore
restructures the simulation as a genuine daily batch loop:

  1. Build the full calendar of all distinct entry-signal dates + exit dates
     across the whole candidate set, iterate it in chronological order.
  2. On each day: first process any exits scheduled to trigger that day
     (releasing concurrency-cap and concentration-cap slots) -- same
     precedence as Run 3's exits-before-entries convention.
  3. Then collect all of that day's NEW entry-eligible candidates (regime=BULL,
     bullish crossover, symbol not already in an open position) into a single
     batch, sort them by conviction score descending, and fill remaining open
     slots (MAX_CONCURRENT - len(open_positions)) top-down, applying the
     MAX_PER_GROUP=2 concentration cap during the ranked fill: a
     high-conviction candidate is skipped (not stopped-on) if its group is
     already at cap, and the NEXT-highest-conviction candidate is tried
     instead.
  4. Any candidate not filled that day is a rejection (either concurrency-cap
     or concentration-cap, both counted) -- there is no queuing/carry-forward
     to a later day, since the original signal (bull_cross on that specific
     historical bar) is a point-in-time event, same as Run 3.

Also tracked: the number of days where the day's batch of new-entry candidates
EXCEEDED the number of open slots -- this is the count of days where ranking
could actually have changed the outcome vs. plain FIFO (on any day where
eligible candidates <= open slots, ranking is a no-op, everything gets in
regardless of order). Reported explicitly per Karan's brief, since a low count
would mean the mechanism rarely engages and any Sharpe delta is noise, not the
ranking mechanism doing real work.

Exit scheme, sizing, charges: BYTE-FOR-BYTE identical to Run 3 (1.5xATR
initial stop, trail arm at +1.0xATR, trail at 2.2xATR, 40-day max-hold,
CNC delivery charges incl. DP charge, ALLOCATED_CAPITAL=250,000,
capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT, ASSUMED_LEVERAGE=1).

Data: OpenAlgo `/history`, daily bars ("D" interval), NSE cash equity,
2021-07-01 -> 2026-06-30, same window as every other backtest this campaign.
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

# ---- Regime / entry (test #8 family, ported to daily bars) -- UNCHANGED FROM RUN 3 ----
REGIME_EMA = 200
ENTRY_FAST, ENTRY_SLOW = 9, 20
ATR_PERIOD = 14

# ---- Exit scheme (replaces 15:00 IST hard exit) -- UNCHANGED FROM RUN 3 ----
INITIAL_STOP_ATR_MULT = 1.5
TRAIL_ARM_ATR_MULT = 1.0     # favourable excursion needed to arm the trail
TRAIL_ATR_MULT = 2.2         # trailing distance once armed, same risk unit as the stop
MAX_HOLD_DAYS = 40           # trading-day backstop if nothing else triggers

LONG_ONLY = True  # CNC cannot short-sell in Indian cash equity

# ---- Sizing (CNC: no leverage) -- UNCHANGED FROM RUN 3 ----
ALLOCATED_CAPITAL = 250_000  # matches this strategy's existing paper-mode capital figure
MAX_CONCURRENT = 6
ASSUMED_LEVERAGE = 1  # CNC requires full payment -- no MIS-style leverage assumption

# ---- Concentration cap -- UNCHANGED FROM RUN 3 ----
MAX_PER_GROUP = 2

# ---- Universe: full Nifty 100, reused verbatim from
# ema_regime_crossover_swing_cnc_nifty100_backtest.py (today's test #4) ----
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BAJAJFINSV", "AXISBANK", "BHARTIARTL",
    "BAJAJ-AUTO", "BAJFINANCE", "BEL", "ASIANPAINT", "DRREDDY", "EICHERMOT",
    "COALINDIA", "CIPLA", "HDFCBANK", "GRASIM", "HCLTECH", "HDFCLIFE", "HINDUNILVR",
    "ICICIBANK", "INFY", "HINDALCO", "INDIGO", "ITC", "JSWSTEEL", "M&M", "KOTAKBANK",
    "LT", "MARUTI", "MAXHEALTH", "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "SBILIFE",
    "SHRIRAMFIN", "SBIN", "RELIANCE", "TATASTEEL", "SUNPHARMA", "TCS", "TECHM",
    "TATACONSUM", "TMPV", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ETERNAL", "JIOFIN",
]

NIFTY_NEXT_50 = [
    "ABB", "ADANIPOWER", "AMBUJACEM", "ADANIENSOL", "ADANIGREEN", "BAJAJHLDNG",
    "BANKBARODA", "CGPOWER", "ZYDUSLIFE", "DLF", "BPCL", "CANBK", "DIVISLAB",
    "BRITANNIA", "CHOLAFIN", "CUMMINSIND", "DMART", "BOSCHLTD", "HAL", "GODREJCP",
    "GAIL", "HDFCAMC", "IOC", "HINDZINC", "INDHOTEL", "JINDALSTEL", "LTM",
    "UNITDSPR", "MUTHOOTFIN", "MOTHERSON", "RECLTD", "PIDILITIND", "PFC", "PNB",
    "TATAPOWER", "SOLARINDS", "SHREECEM", "SIEMENS", "UNIONBANK", "VEDL", "TVSMOTOR",
    "TORNTPHARM", "VBL", "MAZDOCK", "IRFC", "LODHA", "HYUNDAI", "ENRIN", "TATACAP",
    "TMCV",
]

_seen = set()
UNIVERSE = []
for _sym in NIFTY_50 + NIFTY_NEXT_50:
    if _sym not in _seen:
        _seen.add(_sym)
        UNIVERSE.append(_sym)

GROUP_OF = {
    "ADANIENT": "ADANI_GROUP", "ADANIPORTS": "ADANI_GROUP", "ADANIPOWER": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP", "ADANIGREEN": "ADANI_GROUP", "AMBUJACEM": "ADANI_GROUP",
    "TATASTEEL": "TATA_GROUP", "TCS": "TATA_GROUP", "TATACONSUM": "TATA_GROUP",
    "TMPV": "TATA_GROUP", "TITAN": "TATA_GROUP", "TRENT": "TATA_GROUP",
    "TATAPOWER": "TATA_GROUP", "TMCV": "TATA_GROUP", "TATACAP": "TATA_GROUP",
    "INDHOTEL": "TATA_GROUP",
    "BAJAJFINSV": "BAJAJ_GROUP", "BAJAJ-AUTO": "BAJAJ_GROUP", "BAJFINANCE": "BAJAJ_GROUP",
    "BAJAJHLDNG": "BAJAJ_GROUP",
    "GRASIM": "BIRLA_GROUP", "HINDALCO": "BIRLA_GROUP", "ULTRACEMCO": "BIRLA_GROUP",
    "M&M": "MAHINDRA_GROUP", "TECHM": "MAHINDRA_GROUP",
    "VEDL": "VEDANTA_GROUP", "HINDZINC": "VEDANTA_GROUP",
    "LT": "LT_GROUP", "LTM": "LT_GROUP",
    "HDFCBANK": "HDFC_GROUP", "HDFCLIFE": "HDFC_GROUP", "HDFCAMC": "HDFC_GROUP",
    "SBIN": "SBI_GROUP", "SBILIFE": "SBI_GROUP",
    "RELIANCE": "RELIANCE_GROUP", "JIOFIN": "RELIANCE_GROUP",
}
def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ema_regime_crossover_swing_cnc_conviction_ranked")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- CNC (delivery) charges -- reused byte-for-byte from Run 3 ----
BROKERAGE_FLAT = 0.0
BROKERAGE_PCT = 0.0
STT_PCT_DELIVERY = 0.001            # 0.1%, both legs
EXCHANGE_PCT = 0.0000297            # NSE cash-segment rate, product-type-independent
SEBI_PCT = 0.000001
STAMP_DUTY_PCT_DELIVERY = 0.00015   # 0.015% on buy turnover
GST_PCT = 0.18
DP_CHARGE_PER_ISIN = 12.50          # Dhan DP charge, per ISIN per sell-out instruction (verified 2026-07-30)

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


print(f"Universe: {len(UNIVERSE)} names (full Nifty 100, conviction-ranked entry) | "
      f"LONG_ONLY={LONG_ONLY} | product=CNC | capital=Rs{ALLOCATED_CAPITAL:,} | "
      f"max_concurrent={MAX_CONCURRENT} | max_per_group={MAX_PER_GROUP} | "
      f"stop={INITIAL_STOP_ATR_MULT}xATR | trail_arm={TRAIL_ARM_ATR_MULT}xATR | trail={TRAIL_ATR_MULT}xATR | "
      f"max_hold={MAX_HOLD_DAYS}d")

# ------------------------------------------------------------------
# Step 1: per-symbol signal generation, identical logic to test #4,
# except each candidate additionally carries a `conviction` score
# computed at the SIGNAL bar (i.e. df.iloc[i], the bar before the
# t+1 entry), and we ALSO precompute the full exit path (exit_time,
# exit_price, reason, hold_days) exactly as test #4 does, since the
# exit logic itself is completely unchanged -- only WHICH candidates
# get admitted into the portfolio changes.
# ------------------------------------------------------------------
all_candidates = []
skipped = {}

for symbol in UNIVERSE:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    df["ema200"] = df["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    df["regime"] = np.where(df["close"] > df["ema200"], "BULL", "BEAR")
    df["ema_fast"] = df["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df)
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["bear_cross"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])

    print(f"  {symbol}: {len(df):,} daily bars ({df.index[0].date()} -> {df.index[-1].date()})")

    df = df.dropna(subset=["atr", "ema200"])
    n = len(df)
    for i in range(n - 1):
        row = df.iloc[i]
        if not (row["bull_cross"] and row["regime"] == "BULL"):
            continue
        if not LONG_ONLY:
            continue  # (kept for structural symmetry; LONG_ONLY is always True here -- CNC constraint)

        # Conviction score: purely causal, computed at the signal bar (row),
        # using only that bar's own close/ema200/atr14 -- no lookahead.
        atr_signal = row["atr"]
        if atr_signal is None or atr_signal <= 0 or pd.isna(atr_signal):
            continue
        conviction = (row["close"] - row["ema200"]) / atr_signal

        entry_ts = df.index[i + 1]
        entry_bar = df.iloc[i + 1]
        entry_price = entry_bar["open"]
        atr_entry = row["atr"]
        initial_stop = entry_price - INITIAL_STOP_ATR_MULT * atr_entry
        arm_level = entry_price + TRAIL_ARM_ATR_MULT * atr_entry

        trail_stop = initial_stop
        armed = False
        highest_close = entry_price
        exit_px, exit_ts, reason, hold_days = None, None, None, None
        rest = df[df.index > entry_ts]
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
            continue  # ran off the end of history with no exit -- excluded (END_OF_DATA), same convention as Run 3

        pnl_pct = (exit_px - entry_price) / entry_price
        all_candidates.append({
            "symbol": symbol, "direction": "LONG", "entry_time": entry_ts,
            "entry_price": entry_price, "exit_time": exit_ts, "exit_price": exit_px,
            "pnl_pct": pnl_pct, "reason": reason, "hold_days": hold_days,
            "conviction": conviction,
        })

if not all_candidates:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

cand_df = pd.DataFrame(all_candidates).sort_values("entry_time").reset_index(drop=True)
print(f"\nSignal candidates (pre-portfolio-constraints): {len(cand_df)}")

capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT
buying_power = capital_per_trade * ASSUMED_LEVERAGE

# ------------------------------------------------------------------
# Step 2: genuine daily batch simulation. Build the calendar of all
# distinct entry days, process each day in order: exits first, then
# that day's new-entry candidates as a ranked batch (conviction desc)
# against the remaining open slots, applying MAX_PER_GROUP during the
# ranked fill (skip-and-continue to next-highest conviction on cap hit).
# ------------------------------------------------------------------
entry_days = sorted(cand_df["entry_time"].unique())

open_positions = {}       # trade_seq -> {"symbol":..., "group":..., "exit_time":...}
group_open_count = {}
seq = 0

accepted = []
rejected_cap = 0
rejected_group = 0
days_with_more_candidates_than_slots = 0
days_evaluated = 0

# Group candidates by entry day for batch processing.
by_day = {day: g for day, g in cand_df.groupby("entry_time")}

for day in entry_days:
    # --- process exits due today (or earlier, defensively) before evaluating entries ---
    to_close = [seq_id for seq_id, pos in open_positions.items() if pos["exit_time"] <= day]
    for seq_id in to_close:
        pos = open_positions.pop(seq_id)
        group_open_count[pos["group"]] = group_open_count.get(pos["group"], 1) - 1

    day_candidates = by_day[day]
    # Exclude symbols already in an open position (should not normally recur
    # same-day, but defensive per spec: "not already in a position").
    open_symbols = {pos["symbol"] for pos in open_positions.values()}
    eligible = day_candidates[~day_candidates["symbol"].isin(open_symbols)].copy()
    if eligible.empty:
        continue

    days_evaluated += 1
    open_slots = MAX_CONCURRENT - len(open_positions)
    if len(eligible) > max(open_slots, 0):
        days_with_more_candidates_than_slots += 1

    # Rank today's batch by conviction, descending -- highest conviction first.
    eligible = eligible.sort_values("conviction", ascending=False)

    for _, cand in eligible.iterrows():
        if len(open_positions) >= MAX_CONCURRENT:
            rejected_cap += 1
            continue
        grp = group_of(cand["symbol"])
        if group_open_count.get(grp, 0) >= MAX_PER_GROUP:
            rejected_group += 1
            continue  # skip this candidate, try the next-highest-conviction one in the batch

        qty = int(buying_power // cand["entry_price"])
        if qty <= 0:
            continue
        pnl_rupees = qty * cand["entry_price"] * cand["pnl_pct"]

        t = cand.to_dict()
        t["qty"] = qty
        t["pnl_rupees"] = round(pnl_rupees, 2)
        charges = compute_charges_cnc(cand["entry_price"], cand["exit_price"], qty)
        t.update(charges)
        t["net_pnl_rupees"] = round(pnl_rupees - charges["total_charges"], 2)
        t["group"] = grp
        accepted.append(t)

        seq += 1
        open_positions[seq] = {"symbol": cand["symbol"], "group": grp, "exit_time": cand["exit_time"]}
        group_open_count[grp] = group_open_count.get(grp, 0) + 1

result_df = pd.DataFrame(accepted)
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_swing_cnc_conviction_ranked_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | concentration cap ({MAX_PER_GROUP}/group): {rejected_group}")
print(f"Days with >=1 eligible new-entry candidate: {days_evaluated}")
print(f"Days where eligible candidates EXCEEDED open slots (ranking could have mattered): "
      f"{days_with_more_candidates_than_slots} "
      f"({days_with_more_candidates_than_slots / days_evaluated * 100 if days_evaluated else 0:.1f}% of evaluated days)")

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

    print(f"\n=== EMA Regime Crossover -- Swing CNC, Conviction-Ranked Entry "
          f"(Rs{ALLOCATED_CAPITAL:,}, max {MAX_CONCURRENT} concurrent, max {MAX_PER_GROUP}/group) ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Avg holding period: {result_df['hold_days'].mean():.1f} trading days "
          f"(median {result_df['hold_days'].median():.0f})")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"By group (top 10): {result_df['group'].value_counts().head(10).to_dict()}")
    print(f"Avg conviction of accepted trades: {result_df['conviction'].mean():.3f} "
          f"(vs. all candidates: {cand_df['conviction'].mean():.3f})")

    print("\nBy exit reason (net of charges):")
    for reason, g in result_df.groupby("reason"):
        print(f"  {reason:14s} n={len(g):5d} ({len(g)/n*100:4.1f}%)  WR={(g.net_pnl_rupees>0).mean()*100:5.1f}%  "
              f"avg_pnl_pct={g.pnl_pct.mean()*100:+.3f}%  avg_hold={g.hold_days.mean():.1f}d  "
              f"total_net=Rs{g.net_pnl_rupees.sum():+10,.0f}")

    # Daily-return Sharpe: allocate net P&L to its exit date, same convention
    # as every other backtest in this campaign, for direct comparability.
    result_df["exit_time"] = pd.to_datetime(result_df["exit_time"])
    result_df["exit_date"] = result_df["exit_time"].dt.date
    daily_net = result_df.groupby("exit_date")["net_pnl_rupees"].sum() / ALLOCATED_CAPITAL
    daily_gross = result_df.groupby("exit_date")["pnl_rupees"].sum() / ALLOCATED_CAPITAL

    def sharpe(s):
        return s.mean() / s.std(ddof=1) * np.sqrt(252)

    print(f"\nSharpe (gross): {sharpe(daily_gross):.2f}")
    print(f"Sharpe (net)  : {sharpe(daily_net):.2f}   (goal: >= 1.50)")

    dd_df = result_df.sort_values("exit_time").reset_index(drop=True)
    dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
    dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
    dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
    max_dd = dd_df["dd_net"].min()
    print(f"\nMax drawdown (net): Rs {max_dd:,.0f} ({abs(max_dd)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")

    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_swing_cnc_conviction_ranked_trades.csv")
