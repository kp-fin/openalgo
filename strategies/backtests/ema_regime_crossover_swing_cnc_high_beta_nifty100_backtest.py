"""
EMA Regime Crossover -- Swing CNC, High-Beta Top-15 RE-RANKED FROM FULL NIFTY 100 POOL
(2026-08-01, Karan-requested follow-on to the EMA Swing CNC Multi-Day Redesign)

CONTEXT: the deployed EMA Swing CNC strategy (live/paper since 2026-07-31,
`OpenAlgo/strategies/ema_regime_crossover_swing_cnc_signal.py`) trades a
static high-beta 15-name universe ("Run 3",
`ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py`,
Sharpe(net) 2.76 / PF(net) 1.67 / WR(net) 34.9% / maxDD 18.1%, 169 trades).
That 15-name list was ranked by annualized realized-return volatility from a
59-name POOL (test #8's Nifty 50 + a FIXED top-10 Nifty Next 50 subset), not
the full Nifty 100 -- see `ema_regime_crossover_high_beta_rank.py`.

Two adjacent-but-different questions were already asked and answered
separately:
  1. Rolling re-rank of the SAME 59-name pool (monthly/quarterly) -- tested
     2026-07-31, REJECTED (Sharpe(net) ~2.0-2.1 vs static's 2.76).
  2. Trade ALL Nifty 100 names directly (no ranking/shortlist at all) --
     tested 2026-08-01, REJECTED decisively (Sharpe(net) collapsed to 0.55
     via concurrency-cap dilution: more candidates competing for the same
     MAX_CONCURRENT=6 slots).

THIS SCRIPT asks a third, genuinely different question: re-run the EXACT
SAME ranking methodology (`ema_regime_crossover_high_beta_rank.py`'s
causal, no-lookahead annualized-stdev ranking), but from the WIDER
99-name Nifty 100 pool (`ema_regime_crossover_high_beta_rank_nifty100.py`,
run 2026-08-01, all 99 names fetched successfully, no data gaps), still
taking only the TOP 15 by that ranking -- i.e. a same-SIZE list, so the
concurrency-cap dilution mechanism that broke test #2 above does not apply
here. This isolates: did the ORIGINAL 59-name pool happen to exclude
genuinely higher-beta names that would make a BETTER top-15?

RESULT OF THE WIDER-POOL RANKING (see
ema_regime_crossover_high_beta_rank_nifty100/high_beta_ranking_nifty100.csv):
new top 15 = ADANIGREEN, ADANIENSOL, MAZDOCK, MOTHERSON, ADANIENT,
ADANIPOWER, ETERNAL, LODHA, TMCV, IRFC, UNIONBANK, VEDL, ADANIPORTS,
HINDZINC, RECLTD.

  SAME as deployed Run 3 list (9/15): ADANIGREEN, ADANIENSOL, ADANIENT,
  ADANIPOWER, ETERNAL, LODHA, TMCV, VEDL, ADANIPORTS.

  NEW (6/15, not previously considered -- all from the Nifty Next 50 portion
  outside the original 59-pool's fixed top-10 subset): MAZDOCK, MOTHERSON,
  IRFC, UNIONBANK, HINDZINC, RECLTD.

  DROPPED from the deployed list (6/15, now outside the new top15's cutoff):
  CGPOWER, MAXHEALTH, CANBK, SHRIRAMFIN, TRENT, JIOFIN.

EVERYTHING ELSE held byte-for-byte identical to Run 3
(`ema_regime_crossover_swing_cnc_concentration_v3_tightstop_backtest.py`):
daily EMA(200) regime, daily EMA(9)/EMA(20) crossover entry, LONG-only CNC,
1.5xATR initial stop, trailing stop arming at +1.0xATR then trailing at
2.2xATR, 40-trading-day max-hold backstop, MAX_CONCURRENT=6, MAX_PER_GROUP=2,
ASSUMED_LEVERAGE=1, capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT,
ALLOCATED_CAPITAL=250,000, same CNC delivery charges model (incl. DP
charge), same 2021-07-01->2026-06-30 window. ONLY the UNIVERSE/GROUP_OF
constants change (GROUP_OF entries relevant to the 6 new names are pulled
from `ema_regime_crossover_swing_cnc_nifty100_backtest.py`'s already-
extended full-Nifty-100 group map: MAZDOCK and IRFC are deliberately
ungrouped PSU/defense singletons per that script's own documented
rationale, RECLTD/PFC/IRFC PSU financiers are also deliberately NOT grouped
together there -- so of the 6 new names, none actually falls into a
multi-member group; each defaults to its own singleton via `group_of()`'s
fallback, same as Run 3's own non-Adani names).
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

# ---- Universe: top-15 by annualized realized-return volatility, re-ranked
# from the FULL Nifty 100 pool (2026-08-01 ranking run) ----
HIGH_BETA_15_NIFTY100 = [
    "ADANIGREEN", "ADANIENSOL", "MAZDOCK", "MOTHERSON", "ADANIENT",
    "ADANIPOWER", "ETERNAL", "LODHA", "TMCV", "IRFC",
    "UNIONBANK", "VEDL", "ADANIPORTS", "HINDZINC", "RECLTD",
]
UNIVERSE = HIGH_BETA_15_NIFTY100

# Issuer/sector group map -- Adani-group entries reused verbatim from Run 3;
# the 6 new names (MAZDOCK, MOTHERSON, IRFC, UNIONBANK, HINDZINC, RECLTD)
# checked against the full-pool group map in
# ema_regime_crossover_swing_cnc_nifty100_backtest.py: HINDZINC is the only
# one with a documented multi-member group there (VEDANTA_GROUP, paired with
# VEDL, which is also in this universe) -- carried over below. MAZDOCK
# (defense PSU), IRFC/RECLTD (PSU financiers), UNIONBANK (PSU bank), and
# MOTHERSON (standalone) are each deliberately left as singleton groups per
# that script's own rationale (PSU names not grouped as one conglomerate).
GROUP_OF = {
    "ADANIGREEN": "ADANI_GROUP",
    "ADANIENSOL": "ADANI_GROUP",
    "ADANIENT": "ADANI_GROUP",
    "ADANIPOWER": "ADANI_GROUP",
    "ADANIPORTS": "ADANI_GROUP",
    "VEDL": "VEDANTA_GROUP",
    "HINDZINC": "VEDANTA_GROUP",
}
def group_of(symbol):
    return GROUP_OF.get(symbol, f"SINGLE_{symbol}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ema_regime_crossover_swing_cnc_high_beta_nifty100")
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


print(f"Universe: {len(UNIVERSE)} names (high-beta top-15, re-ranked from full Nifty 100 pool) | "
      f"LONG_ONLY={LONG_ONLY} | product=CNC | capital=Rs{ALLOCATED_CAPITAL:,} | "
      f"max_concurrent={MAX_CONCURRENT} | max_per_group={MAX_PER_GROUP} | "
      f"stop={INITIAL_STOP_ATR_MULT}xATR | trail_arm={TRAIL_ARM_ATR_MULT}xATR | trail={TRAIL_ATR_MULT}xATR | "
      f"max_hold={MAX_HOLD_DAYS}d")
all_trades = []
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
            # Ratchet the trailing level using today's completed bar, effective from tomorrow only.
            highest_close = max(highest_close, bar2["close"])
            if not armed and bar2["close"] >= arm_level:
                armed = True
            if armed:
                candidate_trail = highest_close - TRAIL_ATR_MULT * bar2["atr"]
                trail_stop = max(trail_stop, candidate_trail)
        if exit_px is None:
            continue  # ran off the end of history with no exit -- excluded (END_OF_DATA), same convention as Run 3

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
print(f"\nSignal candidates (pre-portfolio-constraints): {len(trades_df)}")

capital_per_trade = ALLOCATED_CAPITAL / MAX_CONCURRENT
buying_power = capital_per_trade * ASSUMED_LEVERAGE

# Chronological event simulation: process exits due before a candidate's entry
# time before evaluating that candidate -- same convention as Run 3.
pending_exits = []
seq = 0
open_positions = {}  # trade_seq -> {"symbol":..., "group":...}
group_open_count = {}

accepted = []
rejected_cap, rejected_group = 0, 0

for _, cand in trades_df.iterrows():
    qty = int(buying_power // cand["entry_price"])
    if qty <= 0:
        continue
    pnl_rupees = qty * cand["entry_price"] * cand["pnl_pct"]

    while pending_exits and pending_exits[0][0] <= cand["entry_time"]:
        _, ex_seq, ex_group = heapq.heappop(pending_exits)
        group_open_count[ex_group] = group_open_count.get(ex_group, 1) - 1
        del open_positions[ex_seq]

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
result_df.to_csv(os.path.join(OUT_DIR, "ema_regime_crossover_swing_cnc_high_beta_nifty100_trades.csv"), index=False)

print(f"Skipped (no data): {skipped}")
print(f"Rejected -- concurrency cap: {rejected_cap} | concentration cap ({MAX_PER_GROUP}/group): {rejected_group}")

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

    print(f"\n=== EMA Regime Crossover -- Swing CNC, High-Beta Top-15 (re-ranked from Nifty 100) "
          f"(Rs{ALLOCATED_CAPITAL:,}, max {MAX_CONCURRENT} concurrent, max {MAX_PER_GROUP}/group) ===")
    print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
    print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
    print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
    print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
    print(f"Avg holding period: {result_df['hold_days'].mean():.1f} trading days "
          f"(median {result_df['hold_days'].median():.0f})")
    print(f"Exit breakdown: {result_df['reason'].value_counts().to_dict()}")
    print(f"By group: {result_df['group'].value_counts().to_dict()}")

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

    print(f"\nTrade log -> {OUT_DIR}/ema_regime_crossover_swing_cnc_high_beta_nifty100_trades.csv")
