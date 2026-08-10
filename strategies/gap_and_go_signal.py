"""
Gap-and-Go with Volume Confirmation — 58-name equity universe — OpenAlgo Forward Test
Run every 5 minutes: 09:15-15:00 IST (Mon-Fri)

DEPLOYED 2026-08-03 as a genuinely different intraday concept from EMA Regime
Crossover (paused same day, "no clear edge" after 17-test + v2 + 1h-timeframe +
slippage campaigns never found a working config). Where EMA Intraday scanned a
static universe every day for a mechanical crossover, this strategy only trades
names with a real catalyst (an overnight gap) AND confirming volume — deliberately
low frequency, high conviction, the one shape this vault's evidence has pointed
toward (the high-beta-universe test on EMA Intraday moved net Sharpe positive only
by cutting trade count; this concept applies that lesson from first principles
instead of retrofitting it onto crossover logic).

*** EVIDENCE STATUS, do not drop this note in future edits: backtest (2021-07-01
to 2026-06-30, 58 of 60 universe names fetched) is the FIRST net-positive result
in this vault's entire intraday-equities campaign — original config: Sharpe(net)
1.10, PF(net) 1.20, WR(net) 47.6%, maxDD(net) 20.8%, both LONG (PF 1.15) and
SHORT (PF 1.36) sides profitable. A walk-forward split held up (Sharpe 1.21 vs
0.98, PF 1.22 vs 1.18, WR 45.7% vs 49.5%).

2026-08-06 UPDATE — LONG gap filter added: backtest analysis showed LONG edge
concentrated in 3–10% gap range (PF 1.43) with near-zero edge outside it (PF
1.04). Filtering LONGs to 3–10% gap (SHORTs unchanged) raises backtest
Sharpe(net) to 1.62, PF to 1.35 on 499 trades — clearing the vault's >=1.5
target. This is still a backtest improvement; the live/paper log remains the
real evidence gate. See
OpenAlgo/strategies/backtests/gap_and_go/exit_research_2026-08-06.md for
the full analysis. equities-system/strategies/gap_and_go.md vault spec
should be updated to reflect this filter change.

Design (matches the backtest exactly, see gap_and_go_backtest.py docstring):
1. Gap filter: |today's open vs. prior day's close| >= GAP_PCT_MIN (1.5%).
2. Volume confirmation: the opening 15m bar's (09:15-09:30 IST) volume must be
   >= VOL_MULT (1.5x) the symbol's own trailing 20-session average volume for
   that same opening bar — computed causally from historical 15m data, no
   lookahead.
3. Direction: gap up -> LONG candidate, gap down -> SHORT candidate. Both
   directions traded (unlike EMA Intraday's SHORT-only convention — the
   long-side weakness found there was specific to that entry logic's regime
   filter, not observed here; backtest shows LONG PF 1.15, SHORT PF 1.36, both
   profitable).
4. Entry: breakout of the opening range (today's 09:15-09:30 bar high/low) in
   the gap direction, confirmed by ENTRY_CUTOFF (10:30 IST). One trade per
   symbol per day, first qualifying breakout only. LIVE-ONLY DIVERGENCE FROM
   THE BACKTEST: the backtest checked breakout against subsequent 15m bar
   highs/lows; this live script checks against live LTP every 5-minute poll —
   tighter fidelity, same accepted-divergence pattern already used in
   ema_regime_crossover_signal.py and orb_spread_signal.py for stop/target
   checks.
5. Stop: opposite side of the opening range. Target: entry +/- 2x opening-range
   width (TARGET_RANGE_MULT). Hard exit 15:00 IST.
6. Sizing: same capital-slice-then-leverage model as every other strategy in
   this vault — POSITION_PCT (10%) of allocated_capital taken FIRST (the real
   margin committed), THEN ASSUMED_MIS_LEVERAGE (5x) applied on top of that
   slice for buying_power. Max MAX_CONCURRENT (6) open positions.
7. Daily loss circuit breaker: halts NEW entries only (never blocks exits) if
   today's realized P&L drops below -DAILY_LOSS_PCT (2%) of allocated_capital —
   same mechanism, same rationale as EMA Intraday's (this vault's own
   isolated-backtest confirmation that this single control does most of the
   drawdown-reduction work applies generically, not just to that strategy).
   No per-symbol same-day loss-block layered on top — the backtest never
   modeled one for this strategy, so the live script doesn't add one either.

Instrument: cash equity, MIS (matches every other strategy in this vault's
`equities-system/`).

Universe: same 60-name candidate list as EMA Intraday's test #8 (full Nifty 50
pool + fixed top-10 Nifty Next 50 subset) — the backtest fetched 58 of 60 (two
names, CGPOWER and ADANIENSOL, hit a known Dhan historical data gap specific to
2021 Q4, which does not affect live/recent data — all 60 are included here).

State persisted to state/gap_and_go_state.json (reset daily at hard exit;
per-symbol daily gap/volume qualification and opening-range levels are
intraday-only, recomputed fresh each trading day).
"""

import json
import logging
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd
import pytz
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capital_state import get_allocated_capital, record_trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gap_and_go")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
IST = pytz.timezone("Asia/Kolkata")

GAP_PCT_MIN = 0.015            # 1.5% minimum overnight gap, matches backtest
VOL_MULT = 1.5                 # opening 15m volume must be >= 1.5x trailing avg
VOL_LOOKBACK_DAYS = 20          # trailing average window for the opening-bar volume baseline
CANDLE_LOOKBACK_DAYS = 45       # comfortably covers 20 trading sessions of history with buffer

MARKET_OPEN = dtime(9, 15)
OPENING_BAR_END = dtime(9, 30)  # first 15m bar: 09:15-09:30, qualification runs once it's completed
ENTRY_CUTOFF = dtime(10, 30)    # opening-range breakout must confirm by this time, matches backtest
HARD_EXIT = dtime(15, 0)
TARGET_RANGE_MULT = 2.0          # target = entry +/- 2x opening-range width, matches backtest

ASSUMED_MIS_LEVERAGE = 5
POSITION_PCT = 0.10
MAX_CONCURRENT = 6
DAILY_LOSS_PCT = 0.02

NIFTY_50 = [
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
UNIVERSE = NIFTY_50 + NEXT50_TOP10

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "gap_and_go_state.json")
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


# ── State helpers ─────────────────────────────────────────────────────────────
def load_state():
    today = str(date.today())
    s = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
        except Exception:
            s = {}

    if s.get("date") != today:
        s = {
            "date": today,
            "positions": {},
            "qualification_done": False,
            "signals": {},   # symbol -> {direction, or_high, or_low, or_width, entered}
            "daily_realized_pnl": 0.0,
            "daily_loss_halted": False,
        }
    s.setdefault("positions", {})
    s.setdefault("qualification_done", False)
    s.setdefault("signals", {})
    s.setdefault("daily_realized_pnl", 0.0)
    s.setdefault("daily_loss_halted", False)
    return s


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, default=str)


# ── OpenAlgo helpers ──────────────────────────────────────────────────────────
def _headers():
    return {"Content-Type": "application/json"}


def get_candles(symbol, interval="15m", lookback_days=CANDLE_LOOKBACK_DAYS):
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.post(f"{HOST}/api/v1/history",
                           json={"apikey": API_KEY, "symbol": symbol, "exchange": "NSE",
                                 "interval": interval, "start_date": start, "end_date": end},
                           headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success" or not data.get("data"):
            log.warning(f"{symbol}: {interval} history fetch failed: {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        log.warning(f"{symbol}: {interval} history fetch error: {e}")
        return pd.DataFrame()


def get_multiquotes(symbols):
    if not symbols:
        return {}
    r = requests.post(f"{HOST}/api/v1/multiquotes",
                       json={"apikey": API_KEY,
                             "symbols": [{"symbol": s, "exchange": "NSE"} for s in symbols]},
                       headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return {res["symbol"]: float(res["data"]["ltp"]) for res in data["results"]}


def _signed_pnl(pos, px):
    return (px - pos["entry_price"]) * pos["qty"] * (1 if pos["direction"] == "LONG" else -1)


def log_closed_trade(sym, pos, exit_px, pnl, reason, now):
    record_trade("gap_and_go", {
        "date": now.strftime("%Y-%m-%d"),
        "symbol": sym,
        "direction": pos["direction"],
        "entry_time": pos["entry_time"],
        "entry_price": round(pos["entry_price"], 2),
        "exit_time": now.isoformat(),
        "exit_price": round(exit_px, 2),
        "qty": pos["qty"],
        "pnl_rupees": round(pnl, 2),
        "reason": reason,
    }, pnl)


def place_order(symbol, action, quantity):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                       json={"apikey": API_KEY, "strategy": "gap_and_go", "symbol": symbol,
                             "exchange": "NSE", "action": action, "quantity": quantity,
                             "pricetype": "MARKET", "product": "MIS"},
                       headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


# ── Qualification (runs once per symbol per day, right after the opening bar) ──
def qualify_symbol(symbol):
    """Returns a signal dict if the symbol clears the gap+volume filter today,
    else None. Matches gap_and_go_backtest.py's generate-candidates logic
    exactly: gap_pct from today's open vs. prior day's close, opening 15m bar
    volume vs. its own trailing 20-session average for that same slot.

    Prior close is the last 15m close before today (equals prior daily close on
    NSE cash when the last bar of the prior session is present)."""
    df15 = get_candles(symbol)
    if df15.empty:
        return None

    today = date.today()
    df15["day"] = df15.index.date
    todays_opening = df15[(df15["day"] == today) & (df15.index.time >= MARKET_OPEN) &
                           (df15.index.time < OPENING_BAR_END)]
    if todays_opening.empty:
        return None
    opening_bar = todays_opening.iloc[0]

    # ponytail: last prior 15m close == daily prior close when prior session's
    # final bar is present; falls back only by rejecting the symbol if missing.
    prior_bars = df15[df15["day"] < today]
    if prior_bars.empty:
        return None
    prior_close = float(prior_bars.iloc[-1]["close"])
    if prior_close <= 0:
        return None
    gap_pct = (float(opening_bar["open"]) - prior_close) / prior_close
    if abs(gap_pct) < GAP_PCT_MIN:
        return None

    # Trailing 20-session average volume for this same opening-bar slot,
    # excluding today, computed causally from historical 15m data.
    hist_opening = df15[(df15["day"] < today) & (df15.index.time >= MARKET_OPEN) &
                         (df15.index.time < OPENING_BAR_END)]
    hist_opening = hist_opening.groupby("day").first().tail(VOL_LOOKBACK_DAYS)
    if len(hist_opening) < VOL_LOOKBACK_DAYS:
        return None  # not enough history yet for a reliable baseline
    vol_avg20 = float(hist_opening["volume"].mean())
    if vol_avg20 <= 0 or float(opening_bar["volume"]) < VOL_MULT * vol_avg20:
        return None

    or_high, or_low = float(opening_bar["high"]), float(opening_bar["low"])
    or_width = or_high - or_low
    if or_width <= 0:
        return None

    direction = "LONG" if gap_pct > 0 else "SHORT"

    # Gap size filter for LONG direction only (backtest analysis 2026-08-06):
    # LONG edge exists only in the 3–10% gap range (PF 1.43 in that band vs
    # PF 1.04 outside it). Filtering LONGs to this range raises backtest
    # Sharpe from 1.10 to 1.62, clearing the vault's >=1.5 target.
    # SHORTs are untouched — their edge is uniform across gap sizes.
    if direction == "LONG" and not (0.03 <= gap_pct <= 0.10):
        log.info(f"SKIP LONG {symbol}: gap={gap_pct*100:+.2f}% outside 3–10% LONG filter")
        return None

    log.info(f"QUALIFIED {symbol}: gap={gap_pct*100:+.2f}% vol={opening_bar['volume']:.0f} "
              f"(avg20={vol_avg20:.0f}) direction={direction} OR=[{or_low:.2f},{or_high:.2f}]")
    return {"direction": direction, "or_high": or_high, "or_low": or_low,
            "or_width": or_width, "entered": False}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(IST)
    t = now.time()
    log.info(f"Gap-and-Go check — {now.strftime('%H:%M:%S IST')}")

    allocated_capital = get_allocated_capital("gap_and_go")
    capital_per_trade = allocated_capital * POSITION_PCT
    buying_power = capital_per_trade * ASSUMED_MIS_LEVERAGE
    daily_loss_limit = DAILY_LOSS_PCT * allocated_capital

    state = load_state()
    positions = state["positions"]
    signals = state["signals"]

    # ── Hard exit: square off everything, stop for the day ───────────────────
    if t >= HARD_EXIT:
        if positions:
            try:
                ltp_map = get_multiquotes(list(positions.keys()))
            except Exception as e:
                log.warning(f"Hard-exit LTP fetch failed: {e}")
                ltp_map = {}
            for sym, pos in list(positions.items()):
                try:
                    exit_action = "SELL" if pos["direction"] == "LONG" else "BUY"
                    place_order(sym, exit_action, pos["qty"])
                    ltp = ltp_map.get(sym)
                    if ltp is None:
                        try:
                            ltp = get_multiquotes([sym]).get(sym)
                        except Exception:
                            ltp = None
                    if ltp is None:
                        log.error(f"HARD EXIT {sym}: order placed but NO price available — "
                                  f"logging P&L as unverified (reason HARD_EXIT_NOPRICE)")
                        log_closed_trade(sym, pos, pos["entry_price"], 0.0, "HARD_EXIT_NOPRICE", now)
                    else:
                        pnl = _signed_pnl(pos, ltp)
                        state["daily_realized_pnl"] += pnl
                        log.info(f"EXIT HARD {pos['direction']} {sym} entry={pos['entry_price']:.2f} exit~={ltp:.2f} pnl~={pnl:+.0f}")
                        log_closed_trade(sym, pos, ltp, pnl, "HARD_EXIT", now)
                    del positions[sym]
                except Exception as e:
                    log.error(f"Hard-exit failed for {sym}: {e}")
            save_state(state)
        return

    # ── Manage open positions: stop/target vs. live LTP ───────────────────────
    if positions:
        try:
            ltp_map = get_multiquotes(list(positions.keys()))
        except Exception as e:
            log.warning(f"Position LTP fetch failed: {e}")
            ltp_map = {}

        for sym, pos in list(positions.items()):
            ltp = ltp_map.get(sym)
            if ltp is None:
                continue
            reason = None
            if pos["direction"] == "LONG":
                if ltp <= pos["stop_px"]:
                    reason = "STOP"
                elif ltp >= pos["target_px"]:
                    reason = "TARGET"
            else:
                if ltp >= pos["stop_px"]:
                    reason = "STOP"
                elif ltp <= pos["target_px"]:
                    reason = "TARGET"

            if reason is None:
                pnl_now = _signed_pnl(pos, ltp)
                log.info(f"HOLD {pos['direction']} {sym} entry={pos['entry_price']:.2f} ltp={ltp:.2f} pnl~={pnl_now:+.0f}")
                continue

            try:
                exit_action = "SELL" if pos["direction"] == "LONG" else "BUY"
                place_order(sym, exit_action, pos["qty"])
                pnl = _signed_pnl(pos, ltp)
                state["daily_realized_pnl"] += pnl
                log.info(f"EXIT {reason} {pos['direction']} {sym} entry={pos['entry_price']:.2f} exit={ltp:.2f} pnl={pnl:+.0f}")
                log_closed_trade(sym, pos, ltp, pnl, reason, now)
                del positions[sym]
            except Exception as e:
                log.error(f"Exit failed for {sym} ({reason}): {e}")

        save_state(state)

    # ── Daily loss circuit breaker: halts new entries only ───────────────────
    if state["daily_realized_pnl"] <= -daily_loss_limit and not state["daily_loss_halted"]:
        state["daily_loss_halted"] = True
        log.warning(f"Daily realized P&L {state['daily_realized_pnl']:+.0f} breached -{daily_loss_limit:.0f} "
                    f"circuit breaker — no new entries for the rest of today. Existing positions still managed normally.")
        save_state(state)
    if state["daily_loss_halted"]:
        return

    # ── Qualification pass: runs once, right after the opening bar completes ──
    if not state["qualification_done"] and t >= OPENING_BAR_END:
        log.info("Running gap+volume qualification pass ...")
        for sym in UNIVERSE:
            sig = qualify_symbol(sym)
            if sig is not None:
                signals[sym] = sig
            time.sleep(1.2)  # stay under Dhan's 1 req/s quote-endpoint cap (DHAN_QUOTE_INTERVAL=1.1s in broker/dhan/api/data.py) — each symbol's daily-history call also triggers a quote call; 0.1s was tripping error 805
        state["qualification_done"] = True
        log.info(f"Qualification complete: {len(signals)} names qualified today")
        save_state(state)

    if t > ENTRY_CUTOFF:
        return  # entry window closed, matches backtest's ENTRY_CUTOFF

    # ── Breakout entries: only for qualified, not-yet-entered symbols ─────────
    pending = {sym: sig for sym, sig in signals.items()
               if not sig["entered"] and sym not in positions}
    if not pending:
        return
    if len(positions) >= MAX_CONCURRENT:
        log.info(f"At {MAX_CONCURRENT}-position cap — no new entries this poll.")
        return

    try:
        ltp_map = get_multiquotes(list(pending.keys()))
    except Exception as e:
        log.warning(f"Breakout-check LTP fetch failed: {e}")
        return

    for sym, sig in pending.items():
        if len(positions) >= MAX_CONCURRENT:
            break
        ltp = ltp_map.get(sym)
        if ltp is None:
            continue

        direction = sig["direction"]
        breakout = (direction == "LONG" and ltp >= sig["or_high"]) or \
                   (direction == "SHORT" and ltp <= sig["or_low"])
        if not breakout:
            continue

        entry_price = ltp
        qty = int(buying_power // entry_price)
        if qty <= 0:
            sig["entered"] = True  # can't size it, don't keep re-checking
            continue

        if direction == "LONG":
            stop_px, target_px = sig["or_low"], entry_price + TARGET_RANGE_MULT * sig["or_width"]
        else:
            stop_px, target_px = sig["or_high"], entry_price - TARGET_RANGE_MULT * sig["or_width"]

        try:
            action = "BUY" if direction == "LONG" else "SELL"
            place_order(sym, action, qty)
            positions[sym] = {
                "direction": direction, "qty": qty, "entry_price": entry_price,
                "stop_px": stop_px, "target_px": target_px, "entry_time": now.isoformat(),
            }
            sig["entered"] = True
            log.info(f"ENTRY {direction} {sym} qty={qty} entry={entry_price:.2f} "
                     f"stop={stop_px:.2f} target={target_px:.2f}")
        except Exception as e:
            log.error(f"Entry failed for {sym} ({direction}): {e}")

    save_state(state)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception:
            log.exception("Unhandled error in main()")
        if datetime.now(IST).time() >= HARD_EXIT:
            log.info("Hard exit reached — stopping")
            break
        time.sleep(300)  # 5 minutes
