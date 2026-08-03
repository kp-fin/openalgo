"""
HH-HL Pullback Breakout (Flag Continuation) -- Nifty 100 Backtest
(2026-08-03, Karan-originated concept, first backtest of this pattern family)

SPEC: see equities-system/strategies/hh_hl_pullback_breakout.md for the full
concept writeup and parameter-decision rationale. Summary:

1. Impulse leg: EMA50 crosses above EMA200 (daily), followed by a ZigZag
   swing-low -> swing-high leg that qualifies as higher-high/higher-low
   structure (checked against the prior confirmed pivots where available).
2. Pullback: price retraces into the 40-60% band of the impulse leg
   (swingH - swingL), then consolidates sideways for >= 5 trading days
   without breaking out above the range or breaking down below it.
3. Breakout: daily close above the running consolidation-range high, with
   volume > 1.5x the 20-day average volume.
4. Entry: next trading day's open (one candle after breakout confirmation).
5. Stop: the consolidation/pullback low (running low since price entered the
   40-60% band).
6. Exit -- two legs: 50% booked at 1:2 risk-reward (risk = entry - stop);
   remaining 50% trails via 2.2xATR once the leg-1 target is hit (same trail
   multiple as the deployed EMA Swing-59/100 strategies, for consistency).

THIS IS A SIGNAL-LEVEL BACKTEST, not portfolio-constrained -- no concurrency
cap, no concentration cap, no position sizing beyond a fixed notional per
trade for P&L-in-rupees reporting. This is idea-stage exploration (first-ever
test of this pattern), matching the vault's convention of reporting raw
signal candidates before layering portfolio constraints (see
ema_regime_crossover_swing_cnc_nifty100_backtest.py's "Signal candidates
(pre-portfolio-constraints)" step for precedent). If the raw signal economics
look promising, a follow-on portfolio-constrained run can be built.

CAUSAL SIMPLIFICATIONS (documented, not hidden):
- ZigZag pivots are causal (a pivot is only used once CONFIRMED, i.e. after
  price has reversed 5% from it) -- standard ZigZag convention, no lookahead.
- "Higher high / higher low" check only compares the impulse leg's swingL/
  swingH against the immediately preceding confirmed pivot of the same type,
  where one exists. If no prior pivot exists (e.g. near the start of the
  history window), the leg is still accepted -- documented weaker case, not
  silently dropped.
- Consolidation window is tracked as a running high/low of (high, low) each
  day from the day price FIRST enters the 40-60% retracement band, so long as
  price does not close back below the impulse leg's swing low (setup
  invalidation) or already break out. The setup is invalidated (state resets,
  waits for a fresh crossover) if price closes below the original impulse
  leg's swingL before a valid breakout occurs.
- No max-hold backstop is defined in the spec for this pattern -- trades that
  run off the end of the history window with neither target nor stop hit are
  excluded (same convention as every other backtest in this vault, e.g. Run
  3's END_OF_DATA exclusion).

UNIVERSE: reused byte-for-byte from
ema_regime_crossover_swing_cnc_nifty100_backtest.py -- union of NIFTY_50 (49
names) + NIFTY_NEXT_50 (50 names) = 99 unique names, sourced 2026-07-17 from
tickertape.in. Same recurring Dhan data-gap names (ADANIENSOL, CGPOWER)
expected to fail non-fatally, same as every prior backtest that includes them.
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

# ---- Pattern parameters, per hh_hl_pullback_breakout.md (all decided 2026-08-03) ----
REGIME_FAST, REGIME_SLOW = 50, 200   # impulse-leg trigger: EMA50 x EMA200 bullish cross
ZIGZAG_PCT = 0.05                    # 5% reversal threshold for swing pivots
RETRACE_LO, RETRACE_HI = 0.40, 0.60  # 40-60% retracement band of the impulse leg
MIN_CONSOLIDATION_DAYS = 5           # minimum sideways days before a breakout counts
VOLUME_AVG_WINDOW = 20
VOLUME_MULT = 1.5                    # breakout-day volume > 1.5x 20d avg volume
DRY_VOLUME_MULT = 0.8                # consolidation-period avg volume must be < 0.8x 20d avg (dry-up)
ATR_PERIOD = 14
LEG1_RR = 2.0                        # leg 1: 50% booked at 1:2 risk-reward
LEG1_FRACTION = 0.5
TRAIL_ATR_MULT = 2.2                 # leg 2 trail multiple (matches EMA Swing-59/100)

NOTIONAL_PER_TRADE = 100_000  # fixed notional for rupee P&L reporting only -- not a sizing model

# ---- Universe: full Nifty 100 = union of NIFTY_50 + NIFTY_NEXT_50 (reused from swing CNC nifty100 run) ----
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hh_hl_pullback_breakout_nifty100")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- CNC (delivery) charges -- reused byte-for-byte rates from Run 3 of the
# EMA Swing CNC campaign (ema_regime_crossover_swing_cnc_nifty100_backtest.py),
# verified against Dhan's real rate card 2026-07-30. Adapted here for a SPLIT
# exit (partial sell at leg 1, remainder at leg 2): stamp duty is charged once
# on the whole buy turnover; STT/exchange/SEBI/GST/DP charge are computed per
# sell instruction (DP charge applies per ISIN per sell-out instruction, so a
# two-leg exit incurs it twice -- this is the correct real-world behaviour,
# not double-counting).
BROKERAGE_FLAT = 0.0
STT_PCT_DELIVERY = 0.001            # 0.1%, both legs
EXCHANGE_PCT = 0.0000297            # NSE cash-segment rate, product-type-independent
SEBI_PCT = 0.000001
STAMP_DUTY_PCT_DELIVERY = 0.00015   # 0.015% on buy turnover
GST_PCT = 0.18
DP_CHARGE_PER_ISIN = 12.50          # Dhan DP charge, per ISIN per sell-out instruction


def compute_charges_cnc_legs(entry_price, qty, sell_legs):
    """sell_legs: list of (exit_price, exit_qty). Returns a dict with the
    same keys as the single-leg model, summed across however many sell
    instructions actually occurred (1 for STOP_BEFORE_LEG1, 2 for TRAIL_STOP)."""
    buy_turnover_total = entry_price * qty
    stamp_duty = STAMP_DUTY_PCT_DELIVERY * buy_turnover_total  # once, on the single buy
    brokerage = stt = exchange_chg = sebi_chg = dp_charge = gst = 0.0
    for exit_price, exit_qty in sell_legs:
        buy_turnover_leg = entry_price * exit_qty
        sell_turnover_leg = exit_price * exit_qty
        total_turnover_leg = buy_turnover_leg + sell_turnover_leg
        leg_brokerage = BROKERAGE_FLAT * 2
        leg_stt = STT_PCT_DELIVERY * total_turnover_leg
        leg_exchange = EXCHANGE_PCT * total_turnover_leg
        leg_sebi = SEBI_PCT * total_turnover_leg
        leg_dp = DP_CHARGE_PER_ISIN
        leg_gst = GST_PCT * (leg_brokerage + leg_exchange + leg_sebi + leg_dp)
        brokerage += leg_brokerage
        stt += leg_stt
        exchange_chg += leg_exchange
        sebi_chg += leg_sebi
        dp_charge += leg_dp
        gst += leg_gst
    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst + dp_charge
    return {"brokerage": round(brokerage, 2), "stt": round(stt, 2), "exchange_chg": round(exchange_chg, 2),
            "sebi_chg": round(sebi_chg, 2), "stamp_duty": round(stamp_duty, 2), "dp_charge": round(dp_charge, 2),
            "gst": round(gst, 2), "total_charges": round(total, 2)}


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


def compute_zigzag(close, pct=ZIGZAG_PCT):
    """Causal ZigZag: returns list of (confirm_i, pivot_i, price, type) in
    confirm-order. A pivot is only appended once price has reversed >= pct
    from the running extreme, so `confirm_i` (when it becomes known) is
    always > `pivot_i` (when it actually occurred)."""
    n = len(close)
    if n < 2:
        return []
    pivots = []
    trend = None  # None until first leg established
    extreme_i, extreme_px = 0, close.iloc[0]
    up_i, up_px = 0, close.iloc[0]      # tentative high before trend known
    down_i, down_px = 0, close.iloc[0]  # tentative low before trend known
    for i in range(1, n):
        px = close.iloc[i]
        if trend is None:
            if px > up_px:
                up_i, up_px = i, px
            if px < down_px:
                down_i, down_px = i, px
            if up_px >= down_px * (1 + pct) and up_i > down_i:
                pivots.append((i, down_i, down_px, "L"))
                trend, extreme_i, extreme_px = "up", up_i, up_px
            elif down_px <= up_px * (1 - pct) and down_i > up_i:
                pivots.append((i, up_i, up_px, "H"))
                trend, extreme_i, extreme_px = "down", down_i, down_px
        elif trend == "up":
            if px > extreme_px:
                extreme_i, extreme_px = i, px
            elif px <= extreme_px * (1 - pct):
                pivots.append((i, extreme_i, extreme_px, "H"))
                trend, extreme_i, extreme_px = "down", i, px
        elif trend == "down":
            if px < extreme_px:
                extreme_i, extreme_px = i, px
            elif px >= extreme_px * (1 + pct):
                pivots.append((i, extreme_i, extreme_px, "L"))
                trend, extreme_i, extreme_px = "up", i, px
    return pivots


print(f"Universe: {len(UNIVERSE)} names (Nifty 100) | ZigZag={ZIGZAG_PCT*100:.0f}% | "
      f"retrace band={RETRACE_LO*100:.0f}-{RETRACE_HI*100:.0f}% | min_consolidation={MIN_CONSOLIDATION_DAYS}d | "
      f"vol_mult={VOLUME_MULT}x | leg1_RR=1:{LEG1_RR:.0f} | trail={TRAIL_ATR_MULT}xATR")

all_trades = []
skipped = {}

for symbol in UNIVERSE:
    df, err = _fetch_daily(symbol, "NSE")
    if err:
        skipped[symbol] = err
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    df["ema_fast"] = df["close"].ewm(span=REGIME_FAST, adjust=False, min_periods=REGIME_FAST).mean()
    df["ema_slow"] = df["close"].ewm(span=REGIME_SLOW, adjust=False, min_periods=REGIME_SLOW).mean()
    df["bull_cross"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["atr"] = compute_atr(df)
    df["vol_avg20"] = df["volume"].rolling(VOLUME_AVG_WINDOW).mean()
    df = df.dropna(subset=["ema_slow"]).reset_index()
    df = df.rename(columns={df.columns[0]: "datetime"})
    n = len(df)
    if n < REGIME_SLOW + 20:
        skipped[symbol] = "insufficient history post-EMA200 warmup"
        print(f"  {symbol}: SKIPPED ({skipped[symbol]})")
        continue

    pivots = compute_zigzag(df["close"])
    # index pivots by confirm_i for fast "what's confirmed by day i" lookups
    pivots_by_confirm = sorted(pivots, key=lambda p: p[0])

    print(f"  {symbol}: {n:,} bars post-warmup, {len(pivots)} zigzag pivots")

    cross_days = df.index[df["bull_cross"]].tolist()

    for cross_i in cross_days:
        # 1) find first confirmed pivot Low at/after cross_i
        confirmed_after = [p for p in pivots_by_confirm if p[1] >= cross_i]
        low_pivots = [p for p in confirmed_after if p[3] == "L"]
        if not low_pivots:
            continue
        confirm_L, idx_L, px_L, _ = low_pivots[0]
        # 2) next confirmed pivot High after that low
        high_pivots = [p for p in confirmed_after if p[3] == "H" and p[1] > idx_L]
        if not high_pivots:
            continue
        confirm_H, idx_H, px_H, _ = high_pivots[0]
        if px_H <= px_L:
            continue

        # 3) HH/HL check against the immediately preceding confirmed pivot of
        # the same type, where one exists (documented weaker case otherwise).
        prior_lows = [p for p in pivots_by_confirm if p[3] == "L" and p[1] < idx_L]
        if prior_lows and px_L < prior_lows[-1][2]:
            continue  # not a higher low
        prior_highs = [p for p in pivots_by_confirm if p[3] == "H" and p[1] < idx_H and p[1] > idx_L]
        # (no earlier high strictly between L and H by construction of zigzag alternation)
        prior_highs_before_L = [p for p in pivots_by_confirm if p[3] == "H" and p[1] < idx_L]
        if prior_highs_before_L and px_H < prior_highs_before_L[-1][2]:
            continue  # not a higher high

        impulse_leg = px_H - px_L
        if impulse_leg <= 0:
            continue
        retrace_hi_px = px_H - RETRACE_LO * impulse_leg  # upper bound of pullback band
        retrace_lo_px = px_H - RETRACE_HI * impulse_leg  # lower bound of pullback band

        # 4) walk forward from confirm_H looking for: band entry -> >=5d
        # consolidation -> volume-confirmed breakout. Invalidate if close
        # drops below px_L (impulse leg's swing low) before breakout.
        band_entry_i = None
        cons_high, cons_low = None, None
        cons_days = 0
        cons_vol_sum = 0.0
        breakout_i = None
        for i in range(confirm_H, n):
            row = df.iloc[i]
            close, high, low, vol = row["close"], row["high"], row["low"], row["volume"]

            if close < px_L:
                break  # setup invalidated

            if band_entry_i is None:
                if retrace_lo_px <= close <= retrace_hi_px:
                    band_entry_i = i
                    cons_high, cons_low = high, low
                    cons_days = 1
                    cons_vol_sum = vol
                continue

            # already in consolidation tracking
            if cons_days >= MIN_CONSOLIDATION_DAYS:
                vol_avg = row["vol_avg20"]
                avg_cons_vol = cons_vol_sum / cons_days  # excludes today's (candidate breakout day) volume
                dry_up_ok = pd.notna(vol_avg) and avg_cons_vol < DRY_VOLUME_MULT * vol_avg
                if pd.notna(vol_avg) and close > cons_high and vol > VOLUME_MULT * vol_avg and dry_up_ok:
                    breakout_i = i
                    break
            cons_high = max(cons_high, high)
            cons_low = min(cons_low, low)
            cons_days += 1
            cons_vol_sum += vol

        if breakout_i is None or breakout_i + 1 >= n:
            continue  # no valid breakout, or breakout on the last available bar (no entry bar)

        entry_i = breakout_i + 1
        entry_ts = df.iloc[entry_i]["datetime"]
        entry_price = df.iloc[entry_i]["open"]
        stop_price = cons_low
        if entry_price <= stop_price:
            continue  # degenerate risk (gapped below stop) -- skip, no valid risk unit
        risk = entry_price - stop_price
        leg1_target = entry_price + LEG1_RR * risk

        # 5) simulate the two-leg exit forward from entry_i+1
        leg1_done = False
        leg1_exit_px, leg1_exit_ts = None, None
        armed = False
        trail_stop = stop_price
        highest_close = entry_price
        final_exit_px, final_exit_ts, final_reason, hold_days = None, None, None, None

        for hold_idx in range(entry_i + 1, n):
            bar = df.iloc[hold_idx]
            ts2 = bar["datetime"]

            if not leg1_done:
                if bar["low"] <= stop_price:
                    # full stop-out before leg 1 target reached -- both legs exit here
                    leg1_exit_px, leg1_exit_ts = stop_price, ts2
                    leg1_done = True
                    final_exit_px, final_exit_ts, final_reason = stop_price, ts2, "STOP_BEFORE_LEG1"
                    hold_days = hold_idx - entry_i
                    break
                if bar["high"] >= leg1_target:
                    leg1_exit_px, leg1_exit_ts = leg1_target, ts2
                    leg1_done = True
                    armed = True
                    trail_stop = leg1_target - TRAIL_ATR_MULT * bar["atr"]
                    highest_close = max(entry_price, bar["close"])
                    continue

            if leg1_done and armed:
                if bar["low"] <= trail_stop:
                    final_exit_px, final_exit_ts, final_reason = trail_stop, ts2, "TRAIL_STOP"
                    hold_days = hold_idx - entry_i
                    break
                highest_close = max(highest_close, bar["close"])
                trail_stop = max(trail_stop, highest_close - TRAIL_ATR_MULT * bar["atr"])

        if final_exit_px is None:
            continue  # ran off end of history before either leg fully resolved -- excluded (END_OF_DATA)

        qty = int(NOTIONAL_PER_TRADE // entry_price)
        if qty <= 0:
            continue

        if final_reason == "STOP_BEFORE_LEG1":
            pnl_pct_blended = (final_exit_px - entry_price) / entry_price
            pnl_rupees_gross = qty * (final_exit_px - entry_price)
            sell_legs = [(final_exit_px, qty)]
        else:
            leg1_qty = qty // 2
            leg2_qty = qty - leg1_qty
            leg1_pnl_pct = (leg1_exit_px - entry_price) / entry_price
            leg2_pnl_pct = (final_exit_px - entry_price) / entry_price
            pnl_pct_blended = LEG1_FRACTION * leg1_pnl_pct + (1 - LEG1_FRACTION) * leg2_pnl_pct
            pnl_rupees_gross = leg1_qty * (leg1_exit_px - entry_price) + leg2_qty * (final_exit_px - entry_price)
            sell_legs = [(leg1_exit_px, leg1_qty), (final_exit_px, leg2_qty)]

        charges = compute_charges_cnc_legs(entry_price, qty, sell_legs)
        pnl_rupees_net = pnl_rupees_gross - charges["total_charges"]

        all_trades.append({
            "symbol": symbol, "cross_date": df.iloc[cross_i]["datetime"], "swingL": px_L, "swingH": px_H,
            "entry_time": entry_ts, "entry_price": entry_price, "stop_price": stop_price,
            "leg1_target": leg1_target, "leg1_exit_price": leg1_exit_px,
            "final_exit_time": final_exit_ts, "final_exit_price": final_exit_px,
            "reason": final_reason, "hold_days": hold_days, "qty": qty,
            "pnl_pct_blended": pnl_pct_blended,
            "pnl_rupees_gross": round(pnl_rupees_gross, 2),
            "total_charges": charges["total_charges"],
            "pnl_rupees_net": round(pnl_rupees_net, 2),
        })

if not all_trades:
    print(f"\nNo trades generated. Skipped: {skipped}")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
trades_df.to_csv(os.path.join(OUT_DIR, "hh_hl_pullback_breakout_nifty100_trades.csv"), index=False)

n = len(trades_df)
wr_gross = (trades_df["pnl_rupees_gross"] > 0).mean() * 100
wr_net = (trades_df["pnl_rupees_net"] > 0).mean() * 100
total_gross = trades_df["pnl_rupees_gross"].sum()
total_charges = trades_df["total_charges"].sum()
total_net = trades_df["pnl_rupees_net"].sum()

gw = trades_df[trades_df["pnl_rupees_gross"] > 0]["pnl_rupees_gross"].sum()
gl = abs(trades_df[trades_df["pnl_rupees_gross"] <= 0]["pnl_rupees_gross"].sum())
pf_gross = gw / gl if gl > 0 else float("inf")
nw = trades_df[trades_df["pnl_rupees_net"] > 0]["pnl_rupees_net"].sum()
nl = abs(trades_df[trades_df["pnl_rupees_net"] <= 0]["pnl_rupees_net"].sum())
pf_net = nw / nl if nl > 0 else float("inf")

print(f"\nSkipped (no data): {skipped}")
print(f"\n=== HH-HL Pullback Breakout -- Nifty 100 (signal-level, ~{NOTIONAL_PER_TRADE:,} notional/trade, CNC charges modelled) ===")
print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
print(f"Gross P&L: Rs {total_gross:+,.0f} | PF (gross): {pf_gross:.2f}")
print(f"Charges  : Rs {total_charges:,.0f}  ({total_charges/total_gross*100 if total_gross else float('nan'):.1f}% of gross P&L)")
print(f"Net P&L  : Rs {total_net:+,.0f} | PF (net): {pf_net:.2f}")
print(f"Avg hold: {trades_df['hold_days'].mean():.1f}d (median {trades_df['hold_days'].median():.0f}d)")
print(f"Exit breakdown: {trades_df['reason'].value_counts().to_dict()}")

dd_df = trades_df.sort_values("entry_time").reset_index(drop=True)
dd_df["cum_pnl_net"] = dd_df["pnl_rupees_net"].cumsum()
dd_df["peak_net"] = dd_df["cum_pnl_net"].cummax()
dd_df["dd_net"] = dd_df["cum_pnl_net"] - dd_df["peak_net"]
print(f"Max drawdown (net, rupees, signal-level): Rs {dd_df['dd_net'].min():,.0f}")

print("\nBy exit reason (net of charges):")
for reason, g in trades_df.groupby("reason"):
    print(f"  {reason:20s} n={len(g):5d} ({len(g)/n*100:4.1f}%)  WR={(g.pnl_rupees_net>0).mean()*100:5.1f}%  "
          f"avg_pnl_pct={g.pnl_pct_blended.mean()*100:+.3f}%  avg_hold={g.hold_days.mean():.1f}d  "
          f"total_net=Rs{g.pnl_rupees_net.sum():+10,.0f}")

print(f"\nTrade log -> {OUT_DIR}/hh_hl_pullback_breakout_nifty100_trades.csv")
