"""
ORB v2 — Peachy Rejection Method — OpenAlgo Forward Test
Run every 5 minutes: 9:45–14:35 IST (Mon–Fri)

Opening Range: 9:15–9:45 IST (30m)
Bearish Reject → BUY ATM PE
Bullish Reject → BUY ATM CE
Lower High     → BUY ATM PE  (confirmation, if no primary)
Higher Low     → BUY ATM CE  (confirmation, if no primary)
Range day skip: price inside OR at 10:15 → no trades today

Target: +40% premium | Stop: -25% premium | Hard exit: 14:30 IST
State persisted to state/orb_v2_state.json (intraday)
"""

import json
import logging
import os
import time
from datetime import date, datetime, time as dtime

import pandas as pd
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("orb_v2")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY     = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST        = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE    = 75
OR_MIN      = 30
OR_MAX      = 150
TARGET_PCT  = 0.40
STOP_PCT    = 0.25
ENTRY_END   = dtime(12, 0)
HARD_EXIT   = dtime(14, 30)
RANGE_CHK   = dtime(10, 15)
IST         = pytz.timezone("Asia/Kolkata")
STATE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "orb_v2_state.json")

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

# ── State helpers ─────────────────────────────────────────────────────────────
def load_state():
    today = str(date.today())
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
            if s.get("date") == today:
                return s
        except Exception:
            pass
    return {"date": today, "orb_high": None, "orb_low": None, "or_locked": False,
            "range_day": False, "bear_traded": False, "bull_traded": False,
            "bear_position": None, "bull_position": None}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

# ── OpenAlgo helpers ──────────────────────────────────────────────────────────
def _headers():
    return {"Content-Type": "application/json"}

def get_candles(interval="5m", lookback_days=1):
    today_str = date.today().isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": interval, "start_date": today_str, "end_date": today_str},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"History API error: {data}")
    df = pd.DataFrame(data["data"])
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("datetime").sort_index()
    return df

def get_ltp(symbol, exchange="NFO"):
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": symbol, "exchange": exchange},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    return float(data["data"]["ltp"])

def get_nifty_spot():
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX"},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])

def get_atm_option_symbol(spot, opt_type):
    """Build ATM option symbol using OpenAlgo optionsymbol endpoint."""
    strike = round(spot / 50) * 50
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO",
                            "expiry": "current", "strike": strike, "optiontype": opt_type},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    return data["data"]["symbol"], strike

def place_order(symbol, action):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": LOT_SIZE,
                            "price_type": "MARKET", "product": "MIS"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"ORB v2 signal check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # ── Build / update 30m OR ─────────────────────────────────────────────────
    if not state["or_locked"]:
        try:
            df = get_candles("5m")
            or_window = df.between_time("09:15", "09:44")
            if not or_window.empty:
                state["orb_high"] = float(or_window["high"].max())
                state["orb_low"]  = float(or_window["low"].min())

            if t >= dtime(9, 45) and state["orb_high"] is not None:
                or_range = state["orb_high"] - state["orb_low"]
                if or_range < OR_MIN or or_range > OR_MAX:
                    log.info(f"OR range {or_range:.0f} pts outside {OR_MIN}–{OR_MAX} — skipping day")
                    state["range_day"] = True
                state["or_locked"] = True
                log.info(f"OR locked: High={state['orb_high']:.2f}  Low={state['orb_low']:.2f}  Range={or_range:.0f}")
        except Exception as e:
            log.warning(f"OR build failed: {e}")
        save_state(state)
        if not state["or_locked"] or state["range_day"]:
            return

    if state["range_day"]:
        log.info("Range day — no trades.")
        return

    orb_high = state["orb_high"]
    orb_low  = state["orb_low"]

    # ── Range day check at 10:15 ──────────────────────────────────────────────
    if t >= RANGE_CHK and not state.get("range_checked"):
        try:
            spot = get_nifty_spot()
            if orb_low < spot < orb_high:
                log.info(f"Range day detected at 10:15 (spot {spot:.2f} inside OR) — no more trades")
                state["range_day"]    = True
                state["range_checked"] = True
                save_state(state)
                return
            state["range_checked"] = True
        except Exception as e:
            log.warning(f"Range day check failed: {e}")

    # ── Exit open positions ───────────────────────────────────────────────────
    for direction in ("bear", "bull"):
        pos = state.get(f"{direction}_position")
        if not pos:
            continue
        try:
            ltp        = get_ltp(pos["symbol"])
            entry_px   = pos["entry_price"]
            pct_change = (ltp - entry_px) / entry_px

            if pct_change >= TARGET_PCT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT TARGET {direction.upper()} {pos['symbol']} — entry {entry_px:.2f} exit {ltp:.2f} (+{pct_change*100:.1f}%)")
                state[f"{direction}_position"] = None
            elif pct_change <= -STOP_PCT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT STOP {direction.upper()} {pos['symbol']} — entry {entry_px:.2f} exit {ltp:.2f} ({pct_change*100:.1f}%)")
                state[f"{direction}_position"] = None
            elif t >= HARD_EXIT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT HARD {direction.upper()} {pos['symbol']} — entry {entry_px:.2f} exit {ltp:.2f} ({pct_change*100:.1f}%)")
                state[f"{direction}_position"] = None
            else:
                log.info(f"HOLD {direction.upper()} {pos['symbol']} — entry {entry_px:.2f} ltp {ltp:.2f} ({pct_change*100:+.1f}%)")
        except Exception as e:
            log.warning(f"Exit check failed for {direction}: {e}")

    save_state(state)

    # ── No new entries after 12:00 or hard exit ───────────────────────────────
    if t > ENTRY_END or t >= HARD_EXIT:
        return

    # ── Signal detection ──────────────────────────────────────────────────────
    try:
        df = get_candles("5m")
        sig = df.between_time("09:45", now.strftime("%H:%M"))
        if len(sig) < 2:
            return

        c  = sig["close"].values
        o  = sig["open"].values
        h  = sig["high"].values
        lo = sig["low"].values
        i  = len(sig) - 1   # current (last complete) candle

        bear_signal = bull_signal = None

        # Primary: Bearish Reject
        if c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
            bear_signal = "BearishReject"

        # Primary: Bullish Reject
        if c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            bull_signal = "BullishReject"

        # Confirmation: Lower High (only if no primary)
        if bear_signal is None and i >= 2:
            if h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
                bear_signal = "LowerHigh"

        # Confirmation: Higher Low (only if no primary)
        if bull_signal is None and i >= 2:
            if lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
                bull_signal = "HigherLow"

        spot = float(c[i])

        # ── Enter Bearish ─────────────────────────────────────────────────────
        if bear_signal and not state["bear_traded"] and not state["bear_position"]:
            try:
                sym, strike = get_atm_option_symbol(spot, "PE")
                entry_ltp   = get_ltp(sym)
                resp        = place_order(sym, "BUY")
                log.info(f"ENTRY BEAR {bear_signal} | {sym} | spot {spot:.2f} | LTP {entry_ltp:.2f}")
                state["bear_traded"]   = True
                state["bear_position"] = {"symbol": sym, "strike": strike, "entry_price": entry_ltp,
                                           "entry_time": now.isoformat(), "signal": bear_signal}
            except Exception as e:
                log.error(f"Bear entry failed: {e}")

        # ── Enter Bullish ─────────────────────────────────────────────────────
        if bull_signal and not state["bull_traded"] and not state["bull_position"]:
            try:
                sym, strike = get_atm_option_symbol(spot, "CE")
                entry_ltp   = get_ltp(sym)
                resp        = place_order(sym, "BUY")
                log.info(f"ENTRY BULL {bull_signal} | {sym} | spot {spot:.2f} | LTP {entry_ltp:.2f}")
                state["bull_traded"]   = True
                state["bull_position"] = {"symbol": sym, "strike": strike, "entry_price": entry_ltp,
                                           "entry_time": now.isoformat(), "signal": bull_signal}
            except Exception as e:
                log.error(f"Bull entry failed: {e}")

    except Exception as e:
        log.error(f"Signal detection failed: {e}")

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
