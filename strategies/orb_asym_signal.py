"""
ORB_Asym — Defensive OR-Hold + asymmetric 1-2-1 butterfly (Sandbox forward test)
Named: ORB_Asym  |  OpenAlgo Python Strategy Host

v2 locked config (2026-08-21 backtest gates: WR 50.6%, Sharpe 12.77):
  Signals: Defensive OR-Hold + quiet prior day (|prev net| <= 0.42%)
  Bull: 3x 5m bars low>=OR_low and close>=OR_mid
  Bear: 3x 5m bars high<=OR_high and close<=OR_mid
  OR: 09:15-09:29, width 30-80; range-day skip at 10:15 (no new entries)
  Legs: BUY ATM x1, SELL OTM1 (body +50) x2, BUY OTM3 (far +150) x1
  Exit: +50 spot target / -25 stop / hard 15:15 — spot-based, matches backtest

Separate from ORB_Spread. Does not import or modify orb_spread_signal.py.
State: state/orb_asym_state.json
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
log = logging.getLogger("orb_asym")

API_KEY = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
STRATEGY = "ORB_Asym"
CAPITAL_KEY = "orb_asym"

LOT_SIZE = 65
FIXED_LOTS = 1  # backtest unit; long legs 1 lot, body short 2 lots
OR_MIN, OR_MAX = 30, 80
HOLD_BARS = 3
TARGET_PTS = 50
STOP_PTS = 25
BODY_OFFSET = "OTM1"   # ±50
FAR_OFFSET = "OTM3"    # ±150
PREV_MOVE_THRESHOLD = 0.42  # % abs — quiet day only (both sides blocked if above)
ENTRY_END = dtime(12, 0)
HARD_EXIT = dtime(15, 15)
RANGE_CHK = dtime(10, 15)
IST = pytz.timezone("Asia/Kolkata")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "orb_asym_state.json")
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
    return {
        "date": today, "orb_high": None, "orb_low": None, "or_locked": False,
        "range_day": False, "range_checked": False,
        "bear_traded": False, "bull_traded": False,
        "bear_position": None, "bull_position": None, "prev_move_pct": None,
    }


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)


def _headers():
    return {"Content-Type": "application/json"}


def get_candles(interval="5m"):
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
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    bar_minutes = int(interval.rstrip("m")) if interval.endswith("m") else None
    if bar_minutes:
        now_naive = datetime.now(IST).replace(tzinfo=None)
        df = df[df.index + pd.Timedelta(minutes=bar_minutes) <= now_naive]
    return df


def get_prev_day_move_pct():
    today = date.today()
    start = (today - timedelta(days=10)).isoformat()
    end = (today - timedelta(days=1)).isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": "D", "start_date": start, "end_date": end},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success" or not data.get("data"):
        raise RuntimeError(f"History API error: {data}")
    last = data["data"][-1]
    return abs(last["close"] - last["open"]) / last["open"] * 100


def get_nifty_spot():
    r = requests.post(f"{HOST}/api/v1/quotes",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX"},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    return float(r.json()["data"]["ltp"])


def get_current_expiry():
    r = requests.post(f"{HOST}/api/v1/expiry",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NFO", "instrumenttype": "options"},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success" or not data.get("data"):
        raise RuntimeError(data)
    return data["data"][0].replace("-", "")


def get_multiquotes(symbols):
    r = requests.post(f"{HOST}/api/v1/multiquotes",
                      json={"apikey": API_KEY,
                            "symbols": [{"symbol": s, "exchange": e} for s, e in symbols]},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return {res["symbol"]: float(res["data"]["ltp"]) for res in data["results"]}


def resolve_option_symbol(option_type, offset, expiry):
    r = requests.post(f"{HOST}/api/v1/optionsymbol",
                      json={"apikey": API_KEY, "underlying": "NIFTY", "exchange": "NSE_INDEX",
                            "expiry_date": expiry, "offset": offset, "option_type": option_type},
                      headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    return data["symbol"]


def place_order(symbol, action, quantity):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "strategy": STRATEGY, "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": quantity,
                            "pricetype": "MARKET", "product": "MIS"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def place_basket(legs):
    orders = [
        {"exchange": "NFO", "symbol": symbol, "action": action, "quantity": quantity,
         "pricetype": "MARKET", "product": "MIS"}
        for symbol, action, quantity in legs
    ]
    r = requests.post(f"{HOST}/api/v1/basketorder",
                      json={"apikey": API_KEY, "strategy": STRATEGY, "orders": orders},
                      headers=_headers(), timeout=25)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Basket order API error: {data}")
    return data.get("results", [])


def _leg_result(results, symbol):
    for r in results:
        if r.get("symbol") == symbol:
            return r
    return None


def resolve_butterfly(option_type):
    """Returns (atm_sym, body_sym, far_sym, net_debit_est, margin_1lot_unit)."""
    expiry = get_current_expiry()
    atm = resolve_option_symbol(option_type, "ATM", expiry)
    body = resolve_option_symbol(option_type, BODY_OFFSET, expiry)
    far = resolve_option_symbol(option_type, FAR_OFFSET, expiry)
    quotes = get_multiquotes([(atm, "NFO"), (body, "NFO"), (far, "NFO")])
    # 1-2-1 debit paid per share of the long unit
    net_debit = quotes[atm] + quotes[far] - 2.0 * quotes[body]
    long_qty = str(LOT_SIZE)
    short_qty = str(2 * LOT_SIZE)
    r = requests.post(f"{HOST}/api/v1/margin",
                      json={"apikey": API_KEY, "positions": [
                          {"exchange": "NFO", "symbol": atm, "action": "BUY",
                           "quantity": long_qty, "product": "MIS", "pricetype": "MARKET"},
                          {"exchange": "NFO", "symbol": body, "action": "SELL",
                           "quantity": short_qty, "product": "MIS", "pricetype": "MARKET"},
                          {"exchange": "NFO", "symbol": far, "action": "BUY",
                           "quantity": long_qty, "product": "MIS", "pricetype": "MARKET"},
                      ]}, headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Margin API error: {data}")
    margin = float(data["data"]["total_margin_required"])
    return atm, body, far, net_debit, margin, quotes


def open_butterfly(atm_sym, body_sym, far_sym):
    """Open 1-2-1. BUY legs before SELL (basket sorts BUY first). Returns entry quotes."""
    long_qty = LOT_SIZE * FIXED_LOTS
    short_qty = 2 * LOT_SIZE * FIXED_LOTS
    results = place_basket([
        (atm_sym, "BUY", long_qty),
        (far_sym, "BUY", long_qty),
        (body_sym, "SELL", short_qty),
    ])
    for sym, label in ((atm_sym, "ATM long"), (far_sym, "far long"), (body_sym, "body short")):
        res = _leg_result(results, sym)
        if not res or res.get("status") != "success":
            raise RuntimeError(f"{label} failed: {res}")
    quotes = get_multiquotes([(atm_sym, "NFO"), (body_sym, "NFO"), (far_sym, "NFO")])
    return quotes


def close_butterfly(pos):
    """Close exact entry symbols. BUY body first, then SELL longs."""
    long_qty = pos["long_qty"]
    short_qty = pos["short_qty"]
    atm, body, far = pos["atm_symbol"], pos["body_symbol"], pos["far_symbol"]
    open_legs = []
    if not pos.get("body_closed"):
        open_legs.append((body, "BUY", short_qty))
    if not pos.get("atm_closed"):
        open_legs.append((atm, "SELL", long_qty))
    if not pos.get("far_closed"):
        open_legs.append((far, "SELL", long_qty))

    if len(open_legs) >= 2:
        results = place_basket(open_legs)
        for sym, action, _qty in open_legs:
            res = _leg_result(results, sym)
            if not res or res.get("status") != "success":
                raise RuntimeError(f"Close {sym} {action} failed: {res}")
            if sym == body:
                pos["body_closed"] = True
            elif sym == atm:
                pos["atm_closed"] = True
            else:
                pos["far_closed"] = True
    else:
        for sym, action, qty in open_legs:
            place_order(sym, action, qty)
            if sym == body:
                pos["body_closed"] = True
            elif sym == atm:
                pos["atm_closed"] = True
            else:
                pos["far_closed"] = True

    return get_multiquotes([(atm, "NFO"), (body, "NFO"), (far, "NFO")])


def butterfly_net(quotes, atm, body, far):
    return quotes[atm] + quotes[far] - 2.0 * quotes[body]


def log_closed_trade(direction, reason, pos, spot, pnl_pts, exit_net, now):
    # P&L per 1-lot unit × FIXED_LOTS: (exit_net - entry_net) * LOT_SIZE
    pnl_rupees = (exit_net - pos["entry_net_debit"]) * LOT_SIZE * FIXED_LOTS
    record_trade(CAPITAL_KEY, {
        "date": now.strftime("%Y-%m-%d"),
        "entry_time": pos["entry_time"],
        "direction": direction,
        "signal": pos["signal"],
        "entry_spot": round(pos["entry_spot"], 2),
        "exit_time": now.isoformat(),
        "exit_spot": round(spot, 2),
        "pnl_pts": round(pnl_pts, 2),
        "entry_net_debit": round(pos["entry_net_debit"], 2),
        "exit_net_debit": round(exit_net, 2),
        "pnl_rupees": round(pnl_rupees, 2),
        "reason": reason,
    }, pnl_rupees)


def can_fit_margin(margin_1unit):
    """Skip if 1 butterfly unit exceeds per-direction capital slice."""
    allocated = get_allocated_capital(CAPITAL_KEY)
    per_side = allocated * 0.40  # same 80%/2 pattern as ORB_Spread
    return margin_1unit <= per_side


def main():
    now = datetime.now(IST)
    t = now.time()
    log.info(f"ORB_Asym check — {now.strftime('%H:%M:%S IST')}")
    state = load_state()

    if state.get("prev_move_pct") is None:
        try:
            state["prev_move_pct"] = get_prev_day_move_pct()
            log.info(f"Prev-day |net| = {state['prev_move_pct']:.2f}% "
                     f"(quiet gate <= {PREV_MOVE_THRESHOLD}%)")
            save_state(state)
        except Exception as e:
            log.warning(f"Prev-day move fetch failed (fail-open): {e}")

    # OR lock: 09:15-09:29 (matches v2 backtest)
    if not state["or_locked"]:
        try:
            df = get_candles("5m")
            or_window = df.between_time("09:15", "09:29")
            if not or_window.empty:
                state["orb_high"] = float(or_window["high"].max())
                state["orb_low"] = float(or_window["low"].min())
            if t >= dtime(9, 45) and state["orb_high"] is not None:
                or_range = state["orb_high"] - state["orb_low"]
                if or_range < OR_MIN or or_range > OR_MAX:
                    log.info(f"OR range {or_range:.0f} outside {OR_MIN}-{OR_MAX} — skip day")
                    state["range_day"] = True
                state["or_locked"] = True
                log.info(f"OR locked High={state['orb_high']:.2f} Low={state['orb_low']:.2f} "
                         f"Range={or_range:.0f}")
        except Exception as e:
            log.warning(f"OR build failed: {e}")
        save_state(state)
        if not state["or_locked"] or state["range_day"]:
            return

    orb_high, orb_low = state["orb_high"], state["orb_low"]
    mid = (orb_high + orb_low) / 2.0

    if t >= RANGE_CHK and not state.get("range_checked"):
        try:
            spot = get_nifty_spot()
            if orb_low < spot < orb_high:
                log.info(f"Range day at 10:15 (spot {spot:.2f}) — no more entries")
                state["range_day"] = True
            state["range_checked"] = True
            save_state(state)
        except Exception as e:
            log.warning(f"Range check failed: {e}")

    # Exits
    for direction, sign in (("bear", 1), ("bull", -1)):
        pos = state.get(f"{direction}_position")
        if not pos:
            continue
        try:
            spot = get_nifty_spot()
            pnl_pts = (pos["entry_spot"] - spot) * sign
            if pnl_pts >= TARGET_PTS or pnl_pts <= -STOP_PTS or t >= HARD_EXIT:
                if pnl_pts >= TARGET_PTS:
                    reason = "TARGET"
                elif pnl_pts <= -STOP_PTS:
                    reason = "STOP"
                else:
                    reason = "HARD_EXIT"
                quotes = close_butterfly(pos)
                exit_net = butterfly_net(quotes, pos["atm_symbol"], pos["body_symbol"], pos["far_symbol"])
                log.info(f"EXIT {reason} {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts) "
                         f"exit_net={exit_net:.2f} entry_net={pos['entry_net_debit']:.2f}")
                log_closed_trade(direction, reason, pos, spot, pnl_pts, exit_net, now)
                state[f"{direction}_position"] = None
            else:
                try:
                    quotes = get_multiquotes([
                        (pos["atm_symbol"], "NFO"), (pos["body_symbol"], "NFO"), (pos["far_symbol"], "NFO")])
                    net_now = butterfly_net(quotes, pos["atm_symbol"], pos["body_symbol"], pos["far_symbol"])
                    log.info(f"HOLD {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts) net={net_now:.2f}")
                except Exception:
                    log.info(f"HOLD {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts)")
        except Exception as e:
            log.warning(f"Exit check failed for {direction}: {e}")
    save_state(state)

    if t > ENTRY_END or t >= HARD_EXIT:
        return
    if state["range_day"]:
        log.info("Range day — no new entries")
        return

    quiet_ok = True
    if state.get("prev_move_pct") is not None and state["prev_move_pct"] > PREV_MOVE_THRESHOLD:
        quiet_ok = False
        log.info(f"Quiet-day gate: prev |net| {state['prev_move_pct']:.2f}% > {PREV_MOVE_THRESHOLD}% "
                 f"— no new entries (ORB_Asym v2)")

    if not quiet_ok:
        return

    try:
        df = get_candles("5m")
        sig = df.between_time("09:45", now.strftime("%H:%M"))
        if len(sig) < HOLD_BARS:
            return
        lows = sig["low"].values
        highs = sig["high"].values
        closes = sig["close"].values
        i = len(sig) - 1

        bull_signal = bear_signal = None
        if all(lows[i - k] >= orb_low for k in range(HOLD_BARS)) and closes[i] >= mid:
            bull_signal = "DefensiveHoldBull"
        if all(highs[i - k] <= orb_high for k in range(HOLD_BARS)) and closes[i] <= mid:
            bear_signal = "DefensiveHoldBear"

        # Bear put butterfly
        if bear_signal and not state["bear_traded"] and not state["bear_position"]:
            try:
                atm, body, far, net_est, margin, _q = resolve_butterfly("PE")
                if not can_fit_margin(margin):
                    log.info(f"BEAR skipped — 1-unit margin Rs {margin:,.0f} exceeds per-side capital")
                else:
                    try:
                        entry_spot = get_nifty_spot()
                    except Exception:
                        entry_spot = float(closes[i])
                    quotes = open_butterfly(atm, body, far)
                    net_debit = butterfly_net(quotes, atm, body, far)
                    log.info(f"ENTRY BEAR PUT BWB {bear_signal} | +{atm} -2x{body} +{far} | "
                             f"entry_spot={entry_spot:.2f} net_debit={net_debit:.2f} "
                             f"(est {net_est:.2f}) margin~{margin:,.0f}")
                    state["bear_traded"] = True
                    state["bear_position"] = {
                        "atm_symbol": atm, "body_symbol": body, "far_symbol": far,
                        "long_qty": LOT_SIZE * FIXED_LOTS, "short_qty": 2 * LOT_SIZE * FIXED_LOTS,
                        "entry_spot": entry_spot, "entry_net_debit": net_debit,
                        "entry_time": now.isoformat(), "signal": bear_signal,
                    }
            except Exception as e:
                log.error(f"Bear entry failed: {e}")

        # Bull call butterfly
        if bull_signal and not state["bull_traded"] and not state["bull_position"]:
            try:
                atm, body, far, net_est, margin, _q = resolve_butterfly("CE")
                if not can_fit_margin(margin):
                    log.info(f"BULL skipped — 1-unit margin Rs {margin:,.0f} exceeds per-side capital")
                else:
                    try:
                        entry_spot = get_nifty_spot()
                    except Exception:
                        entry_spot = float(closes[i])
                    quotes = open_butterfly(atm, body, far)
                    net_debit = butterfly_net(quotes, atm, body, far)
                    log.info(f"ENTRY BULL CALL BWB {bull_signal} | +{atm} -2x{body} +{far} | "
                             f"entry_spot={entry_spot:.2f} net_debit={net_debit:.2f} "
                             f"(est {net_est:.2f}) margin~{margin:,.0f}")
                    state["bull_traded"] = True
                    state["bull_position"] = {
                        "atm_symbol": atm, "body_symbol": body, "far_symbol": far,
                        "long_qty": LOT_SIZE * FIXED_LOTS, "short_qty": 2 * LOT_SIZE * FIXED_LOTS,
                        "entry_spot": entry_spot, "entry_net_debit": net_debit,
                        "entry_time": now.isoformat(), "signal": bull_signal,
                    }
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
        time.sleep(300)
