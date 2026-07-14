"""
ORB v1 — Opening Range Breakout — OpenAlgo Forward Test
Run every 15 minutes: 9:30–14:35 IST (Mon–Fri)

Opening Range: 9:15–9:30 IST (15m)
Direction: SHORT only → BUY ATM PE on breakout below OR low
Entry window: 9:30–12:00 IST | Hard exit: 14:30 IST
Target: +35% premium | Stop: −25% premium
State: state/orb_v1_state.json
"""

import json
import logging
import os
import time
from datetime import date, datetime, time as dtime

import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("orb_v1")

API_KEY    = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST       = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE   = 75
TARGET_PCT = 0.35
STOP_PCT   = 0.25
ENTRY_END  = dtime(12, 0)
HARD_EXIT  = dtime(14, 30)
IST        = pytz.timezone("Asia/Kolkata")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "orb_v1_state.json")

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


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
            "traded": False, "position": None}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

def _h():
    return {"Content-Type": "application/json"}

def get_candles_today(interval="15m"):
    today = date.today().isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": interval, "start_date": today, "end_date": today},
                      headers=_h(), timeout=15)
    r.raise_for_status()
    import pandas as pd
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    df = pd.DataFrame(data["data"])
    df["datetime"] = pd.to_datetime(df["timestamp"])
    return df.set_index("datetime").sort_index()

def get_ltp(symbol, exchange="NFO"):
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": symbol, "exchange": exchange},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])

def get_nifty_spot():
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX"},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])

def get_atm_pe_symbol(spot):
    strike = round(spot / 50) * 50
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO",
                            "expiry": "current", "strike": strike, "optiontype": "PE"},
                      headers=_h(), timeout=10)
    r.raise_for_status()
    return r.json()["data"]["symbol"], strike

def place_order(symbol, action):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": LOT_SIZE,
                            "price_type": "MARKET", "product": "MIS"},
                      headers=_h(), timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"ORB v1 check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # Lock OR after 9:30
    if not state["or_locked"] and t >= dtime(9, 30):
        try:
            df = get_candles_today("5m")
            or_window = df.between_time("09:15", "09:29")
            if not or_window.empty:
                state["orb_high"] = float(or_window["high"].max())
                state["orb_low"]  = float(or_window["low"].min())
                state["or_locked"] = True
                log.info(f"OR locked: H={state['orb_high']:.2f} L={state['orb_low']:.2f}")
        except Exception as e:
            log.warning(f"OR lock failed: {e}")
        save_state(state)

    if not state["or_locked"]:
        return

    orb_low = state["orb_low"]

    # Exit open position
    if state["position"]:
        pos = state["position"]
        try:
            ltp    = get_ltp(pos["symbol"])
            entry  = pos["entry_price"]
            pct    = (ltp - entry) / entry

            if pct >= TARGET_PCT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT TARGET {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} (+{pct*100:.1f}%)")
                state["position"] = None
            elif pct <= -STOP_PCT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT STOP {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} ({pct*100:.1f}%)")
                state["position"] = None
            elif t >= HARD_EXIT:
                place_order(pos["symbol"], "SELL")
                log.info(f"EXIT HARD {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} ({pct*100:.1f}%)")
                state["position"] = None
            else:
                log.info(f"HOLD {pos['symbol']} entry={entry:.2f} ltp={ltp:.2f} ({pct*100:+.1f}%)")
        except Exception as e:
            log.warning(f"Exit check failed: {e}")
        save_state(state)

    if t > ENTRY_END or t >= HARD_EXIT or state["traded"]:
        return

    # Signal: bearish close below OR low (SHORT proxy)
    try:
        spot = get_nifty_spot()
        if spot < orb_low:
            sym, strike = get_atm_pe_symbol(spot)
            entry_ltp   = get_ltp(sym)
            place_order(sym, "BUY")
            log.info(f"ENTRY SHORT (ORB breakout) | {sym} | spot={spot:.2f} | LTP={entry_ltp:.2f}")
            state["traded"]   = True
            state["position"] = {"symbol": sym, "strike": strike, "entry_price": entry_ltp,
                                  "entry_time": now.isoformat()}
    except Exception as e:
        log.error(f"Entry failed: {e}")

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
        time.sleep(900)  # 15 minutes
