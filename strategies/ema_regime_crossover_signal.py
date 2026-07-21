"""
EMA Regime Crossover — 20-name equity universe — OpenAlgo Forward Test
Run every 5 minutes: 09:15-15:00 IST (Mon-Fri)

Regime (30m, EMA 200): price above EMA200 on 30m -> BULL regime, only
longs permitted. Below -> BEAR regime, only shorts permitted. 30m bars
are derived by resampling fetched 15m bars (OpenAlgo /api/v1/history has
no native 30m interval) -- matches the backtest exactly.

Entry (15m, EMA9/EMA20 crossover): bullish cross (9 crosses above 20)
while in BULL regime -> LONG. Bearish cross (9 crosses below 20) while
in BEAR regime -> SHORT. Crossovers against the current regime are not
traded. Regime flip does NOT force-close an open position -- it only
blocks new entries in the direction no longer in regime; an open
position rides its own exit rule to completion.

Exit: 1.36xATR(14) stop / 3xATR(14) target on the 15m entry chart, or an
opposite EMA9/EMA20 crossover, whichever comes first. Hard exit 15:00
IST (EOD square-off, matches backtest -- this strategy is intraday
only, not multi-day). Unlike the backtest (which checks stop/target
against 15m bar high/low), this live script checks stop/target against
LIVE LTP every 5-minute poll -- tighter fidelity than a 15m bar close,
same accepted-divergence pattern already used in orb_spread_signal.py.
Reverse-crossover exit is still checked from the most recently CLOSED
15m bar only (causal, matches backtest for that specific trigger).

Instrument: cash equity, MIS (Karan's call, 2026-07-17 -- matches how
the backtest was computed on spot returns, works across the whole
universe since not all names are F&O-eligible).

Capital: allocated_capital starts at Rs 15,000 (Karan's original account size,
confirmed 2026-07-17) and is read fresh from capital_state.py's
capital_allocation.json each poll -- NOT a fixed module constant. As of
2026-07-18 (Karan's reinvestment rule), live-mode closed trades compound
their P&L into allocated_capital directly (see capital_state.record_trade);
paper-mode trades never touch it. Position sizing uses leveraged buying_power
(allocated_capital x ASSUMED_MIS_LEVERAGE), not raw own capital -- MIS
intraday equity gives "up to 5x" leverage per Karan, though it varies
stock-to-stock. 5x is used here as a sizing-planning CEILING, not a precise
per-stock figure -- real Dhan MIS margin at order time is the actual final
gate; an order that fails margin validation is caught and logged, not
treated as fatal. Sizing against raw capital (no leverage) made 5 of 20
universe names completely untradeable and 10 more coarse 1-share-only
positions at the original Rs 15,000 -- see ema_regime_crossover.md's
"Open Item: MTF / Swing Capital Model" section for the numbers that drove
this.

Daily loss circuit breaker is sized against allocated_capital (real money at
risk), NOT the leveraged buying power -- 2% of current allocated_capital
(Karan confirmed the 2% figure and the original Rs 15,000 base 2026-07-17;
the 2% ratio itself doesn't change as capital compounds).

Sizing (corrected 2026-07-21, Karan-confirmed -- replaces the same day's
earlier, wrong version): POSITION_PCT (10%) is taken from allocated_capital
FIRST -- capital_per_trade = allocated_capital x 10% (Rs 25,000 at the
current Rs 2,50,000 base) -- THEN leverage is applied on top of that slice:
buying_power = capital_per_trade x ASSUMED_MIS_LEVERAGE (Rs 1,25,000 at 5x).
The FULL resulting buying_power is used for qty, no further cap layered on.
This is the correct order: capital_per_trade is the real margin committed to
the trade (leverage doesn't change how much of your own money is at risk,
only how big a position that money can control). The same-day earlier
version got this backwards -- it computed buying_power from the FULL
allocated_capital first, then tried to shrink the leveraged notional back
down (12.5% of buying_power, then a flat Rs 25,000 cap on that already-
leveraged number) -- which meant only Rs 25,000/leverage = Rs 5,000 of real
capital was actually being committed per trade, not Rs 25,000. Max 6
concurrent positions across the whole universe; skip a new signal beyond
that cap. At 6 positions x Rs 25,000 capital_per_trade, total margin
committed is capped at Rs 1,50,000 (60% of allocated capital).

Per-symbol position limit (LIVE-ONLY DIVERGENCE FROM THE BACKTEST):
the backtest's signal-generation pass does not prevent two independent
"trades" from overlapping in the same symbol -- it only caps GLOBAL
concurrency at 6, not per-symbol. A real account can't hold two
separately-tracked entries in one name the same way, so this live
script adds an implicit per-symbol cap of 1 (no new entry in a symbol
that already has an open position) on top of the 6-position global
cap. This is a deliberate, small fidelity gap vs. the backtest --
flagged in ema_regime_crossover.md, not silently introduced.

Daily loss circuit breaker: halts NEW entries only (never blocks exits)
if today's realized P&L drops below -daily_loss_limit (2% of the current
allocated_capital, recomputed each poll -- was a fixed Rs 300 before
2026-07-18's compounding rule; both the 2% figure and the original Rs 15,000
base are Karan's own confirmed numbers, 2026-07-17, not an assumption).

Universe: top 10 Nifty 50 + top 10 Nifty Next 50 by trailing 20-day
average daily traded value, re-ranked on the first poll of each new
calendar month (confirmed monthly cadence, 2026-07-17). The CANDIDATE
POOL itself (the ~49+49 names below) is held static -- re-ranking picks
top 10 from WITHIN this fixed pool, it does not re-fetch actual NSE
index membership (no clean source for that exists in this vault, per
universe_ranking.py's own note). A name dropping out of the monthly
top-10 blocks new entries only -- an existing open position rides to
its own exit, matching the regime-flip precedent (confirmed 2026-07-17).

State persisted to state/ema_regime_crossover_state.json (intraday
positions/P&L reset daily at hard exit; universe/month persists).
"""

import json
import logging
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests

# Deployed copies run from strategies/scripts/ (Python Strategy Host), where
# only the script's own directory is on sys.path by default -- add the parent
# strategies/ dir so capital_state.py resolves from both source and deployed
# locations without needing a duplicated copy kept in sync.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capital_state import get_allocated_capital, record_trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ema_regime_crossover")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST      = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
IST       = pytz.timezone("Asia/Kolkata")

REGIME_EMA       = 200
ENTRY_FAST       = 9
ENTRY_SLOW       = 20
ATR_PERIOD       = 14
STOP_ATR_MULT    = 1.36
TARGET_ATR_MULT  = 3.0
HARD_EXIT        = dtime(15, 0)
CANDLE_LOOKBACK_DAYS = 60   # comfortably covers 200x30m-bar EMA warmup (~16 trading days) with buffer

ASSUMED_MIS_LEVERAGE = 5          # "up to 5x", varies stock-to-stock per Karan -- planning ceiling only,
                                   # real Dhan MIS margin at order time is the actual gate (see module docstring)
POSITION_PCT         = 0.10       # of allocated_capital, taken BEFORE leverage -- this is the real capital/margin
                                   # committed per trade (Rs 25,000 at Rs 2,50,000 capital), corrected 2026-07-21
MAX_CONCURRENT       = 6
DAILY_LOSS_PCT       = 0.02       # of allocated_capital -- sized against real capital at risk, not leveraged buying power

# OWN_CAPITAL/BUYING_POWER/DAILY_LOSS_LIMIT used to be fixed constants (Rs
# 15,000 / Rs 75,000 / Rs 300). As of 2026-07-18 (Karan's capital-reinvestment
# rule) allocated capital compounds with live P&L, so these are now read fresh
# from capital_state.py each poll in main() instead of computed once at import.

UNIVERSE_EACH = 10

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

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "ema_regime_crossover_state.json")
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


# ── State helpers ─────────────────────────────────────────────────────────────
def load_state():
    today = str(date.today())
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
        except Exception:
            s = {}
    else:
        s = {}

    if s.get("date") != today:
        # New trading day: flatten intraday state, keep universe/month.
        s = {
            "date": today,
            "universe_month": s.get("universe_month"),
            "universe": s.get("universe", []),
            "positions": {},
            "daily_realized_pnl": 0.0,
            "daily_loss_halted": False,
        }
    s.setdefault("positions", {})
    s.setdefault("universe", [])
    s.setdefault("universe_month", None)
    s.setdefault("daily_realized_pnl", 0.0)
    s.setdefault("daily_loss_halted", False)
    return s


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)


# ── OpenAlgo helpers ──────────────────────────────────────────────────────────
def _headers():
    return {"Content-Type": "application/json"}


def get_candles(symbol, lookback_days=CANDLE_LOOKBACK_DAYS):
    """15m candles, lookback_days of history. Returns empty df on failure
    (caller decides whether that's fatal for this symbol this poll)."""
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.post(f"{HOST}/api/v1/history",
                          json={"apikey": API_KEY, "symbol": symbol, "exchange": "NSE",
                                "interval": "15m", "start_date": start, "end_date": end},
                          headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success" or not data.get("data"):
            log.warning(f"{symbol}: history fetch failed: {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.set_index("datetime").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        log.warning(f"{symbol}: history fetch error: {e}")
        return pd.DataFrame()


def get_multiquotes(symbols):
    """symbols: list of symbol strings (all NSE). Returns {symbol: ltp}."""
    r = requests.post(f"{HOST}/api/v1/multiquotes",
                      json={"apikey": API_KEY,
                            "symbols": [{"symbol": s, "exchange": "NSE"} for s in symbols]},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return {res["symbol"]: float(res["data"]["ltp"]) for res in data["results"]}


def log_closed_trade(sym, pos, exit_px, pnl, reason, now):
    """Append to the trade log (paper AND live) for ongoing Sharpe/PF/win-rate/
    avg-P&L tracking (2026-07-18). Live mode also compounds pnl into
    allocated_capital -- see capital_state.py."""
    record_trade("ema_regime_crossover", {
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
                      json={"apikey": API_KEY, "strategy": "ema_regime_crossover", "symbol": symbol,
                            "exchange": "NSE", "action": action, "quantity": quantity,
                            "pricetype": "MARKET", "product": "MIS"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def avg_daily_traded_value(symbol, lookback_days=20):
    df = get_candles_daily(symbol)
    if df is None or df.empty or "volume" not in df.columns:
        return None
    df = df.tail(lookback_days)
    if df.empty:
        return None
    return float((df["close"] * df["volume"]).mean())


def get_candles_daily(symbol, lookback_days=45):
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.post(f"{HOST}/api/v1/history",
                          json={"apikey": API_KEY, "symbol": symbol, "exchange": "NSE",
                                "interval": "D", "start_date": start, "end_date": end},
                          headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success" or not data.get("data"):
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        log.warning(f"{symbol}: daily history fetch error: {e}")
        return pd.DataFrame()


def refresh_universe():
    """Rank both candidate pools by trailing 20-day ADV, take top 10 each.
    Only the RANKING is recomputed monthly -- the candidate pools
    (NIFTY_50 / NIFTY_NEXT_50 above) are a static snapshot, see module
    docstring."""
    log.info("Universe refresh starting (monthly) ...")
    ranked = {}
    for pool_name, pool in (("Nifty 50", NIFTY_50), ("Nifty Next 50", NIFTY_NEXT_50)):
        rows = []
        for sym in pool:
            adv = avg_daily_traded_value(sym)
            if adv is not None:
                rows.append((sym, adv))
            time.sleep(0.1)
        rows.sort(key=lambda x: x[1], reverse=True)
        top = [sym for sym, _ in rows[:UNIVERSE_EACH]]
        ranked[pool_name] = top
        log.info(f"{pool_name} top {UNIVERSE_EACH}: {top}")
    universe = ranked["Nifty 50"] + ranked["Nifty Next 50"]
    log.info(f"Universe refreshed: {len(universe)} names")
    return universe


# ── Indicator helpers ────────────────────────────────────────────────────────
def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_indicators(df15):
    """Adds regime (from resampled 30m EMA200), entry EMA9/20, ATR14, and
    bull/bear crossover flags to a 15m dataframe. Returns None if there's
    not enough history for a valid EMA200 regime reading yet."""
    if df15.empty or len(df15) < ENTRY_SLOW + 2:
        return None

    df30 = df15.resample("30min", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["open"])
    df30["ema200"] = df30["close"].ewm(span=REGIME_EMA, adjust=False, min_periods=REGIME_EMA).mean()
    if df30["ema200"].dropna().empty:
        return None  # not enough history yet for a valid regime reading
    df30["regime"] = np.where(df30["close"] > df30["ema200"], "BULL", "BEAR")

    df15 = df15.copy()
    df15["ema_fast"] = df15["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
    df15["ema_slow"] = df15["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
    df15["atr"] = compute_atr(df15)
    df15["bull_cross"] = (df15["ema_fast"].shift(1) <= df15["ema_slow"].shift(1)) & (df15["ema_fast"] > df15["ema_slow"])
    df15["bear_cross"] = (df15["ema_fast"].shift(1) >= df15["ema_slow"].shift(1)) & (df15["ema_fast"] < df15["ema_slow"])
    df15["regime"] = df30["regime"].reindex(df15.index, method="ffill")
    return df15.dropna(subset=["atr", "regime"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"EMA Regime Crossover check — {now.strftime('%H:%M:%S IST')}")

    # Read fresh each poll -- allocated_capital compounds with live P&L
    # (2026-07-18), so this can't be a module-level constant anymore.
    allocated_capital = get_allocated_capital("ema_regime_crossover")
    capital_per_trade  = allocated_capital * POSITION_PCT   # real margin per trade -- see module docstring "Sizing"
    buying_power       = capital_per_trade * ASSUMED_MIS_LEVERAGE
    daily_loss_limit   = DAILY_LOSS_PCT * allocated_capital

    state = load_state()

    # ── Monthly universe refresh (first poll of a new calendar month) ────────
    this_month = now.strftime("%Y-%m")
    if state["universe_month"] != this_month:
        try:
            state["universe"] = refresh_universe()
            state["universe_month"] = this_month
            save_state(state)
        except Exception as e:
            log.error(f"Universe refresh failed, keeping previous universe: {e}")
            if not state["universe"]:
                log.error("No universe available at all — cannot proceed this poll.")
                return

    universe = state["universe"]
    positions = state["positions"]

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
                    ltp = ltp_map.get(sym, pos["entry_price"])
                    pnl = (ltp - pos["entry_price"]) * pos["qty"] * (1 if pos["direction"] == "LONG" else -1)
                    state["daily_realized_pnl"] += pnl
                    log.info(f"EXIT HARD {pos['direction']} {sym} entry={pos['entry_price']:.2f} exit~={ltp:.2f} pnl~={pnl:+.0f}")
                    log_closed_trade(sym, pos, ltp, pnl, "HARD_EXIT", now)
                    del positions[sym]
                except Exception as e:
                    log.error(f"Hard-exit failed for {sym}: {e}")
            save_state(state)
        return

    # ── Symbols needing a candle fetch this poll: universe + any open position
    #    not currently in the universe (dropped names still need reverse-cross
    #    exit monitoring) ──────────────────────────────────────────────────────
    symbols_to_check = sorted(set(universe) | set(positions.keys()))
    indicators = {}
    for sym in symbols_to_check:
        df15 = get_candles(sym)
        ind = compute_indicators(df15)
        if ind is not None:
            indicators[sym] = ind
        time.sleep(0.1)

    # ── Manage open positions: ATR stop/target (live LTP) + reverse-cross
    #    (bar-close) + hard exit is handled above ─────────────────────────────
    if positions:
        try:
            ltp_map = get_multiquotes(list(positions.keys()))
        except Exception as e:
            log.warning(f"Position LTP fetch failed: {e}")
            ltp_map = {}

        for sym, pos in list(positions.items()):
            ltp = ltp_map.get(sym)
            reason = None

            if ltp is not None:
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

            if reason is None and sym in indicators:
                last = indicators[sym].iloc[-1]
                if pos["direction"] == "LONG" and bool(last["bear_cross"]):
                    reason = "REVERSE_CROSS"
                elif pos["direction"] == "SHORT" and bool(last["bull_cross"]):
                    reason = "REVERSE_CROSS"

            if reason is None:
                if ltp is not None:
                    pnl_now = (ltp - pos["entry_price"]) * pos["qty"] * (1 if pos["direction"] == "LONG" else -1)
                    log.info(f"HOLD {pos['direction']} {sym} entry={pos['entry_price']:.2f} ltp={ltp:.2f} pnl~={pnl_now:+.0f}")
                continue

            try:
                exit_action = "SELL" if pos["direction"] == "LONG" else "BUY"
                place_order(sym, exit_action, pos["qty"])
                exit_px = ltp if ltp is not None else pos["entry_price"]
                pnl = (exit_px - pos["entry_price"]) * pos["qty"] * (1 if pos["direction"] == "LONG" else -1)
                state["daily_realized_pnl"] += pnl
                log.info(f"EXIT {reason} {pos['direction']} {sym} entry={pos['entry_price']:.2f} exit={exit_px:.2f} pnl={pnl:+.0f}")
                log_closed_trade(sym, pos, exit_px, pnl, reason, now)
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

    # ── New entries: regime + crossover aligned, no existing position in the
    #    symbol, under the global concurrency cap ────────────────────────────
    if len(positions) >= MAX_CONCURRENT:
        log.info(f"At {MAX_CONCURRENT}-position cap — no new entries this poll.")
        return

    for sym in universe:
        if len(positions) >= MAX_CONCURRENT:
            break
        if sym in positions:
            continue
        ind = indicators.get(sym)
        if ind is None or ind.empty:
            continue
        last = ind.iloc[-1]

        direction = None
        if bool(last["bull_cross"]) and last["regime"] == "BULL":
            direction = "LONG"
        elif bool(last["bear_cross"]) and last["regime"] == "BEAR":
            direction = "SHORT"
        if direction is None:
            continue

        entry_price = float(last["close"])
        atr = float(last["atr"])
        if atr <= 0 or np.isnan(atr):
            continue
        qty = int(buying_power // entry_price)
        if qty <= 0:
            continue

        if direction == "LONG":
            stop_px, target_px = entry_price - STOP_ATR_MULT * atr, entry_price + TARGET_ATR_MULT * atr
        else:
            stop_px, target_px = entry_price + STOP_ATR_MULT * atr, entry_price - TARGET_ATR_MULT * atr

        try:
            action = "BUY" if direction == "LONG" else "SELL"
            place_order(sym, action, qty)
            positions[sym] = {
                "direction": direction, "qty": qty, "entry_price": entry_price,
                "stop_px": stop_px, "target_px": target_px, "entry_time": now.isoformat(),
            }
            log.info(f"ENTRY {direction} {sym} qty={qty} entry={entry_price:.2f} "
                     f"stop={stop_px:.2f} target={target_px:.2f} atr={atr:.2f}")
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
