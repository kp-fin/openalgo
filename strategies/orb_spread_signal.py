"""
ORB_Spread — Peachy Rejection Method, Debit Spread Execution — OpenAlgo Forward Test
(formerly ORB v2, then ORB_kp — renamed ORB_Spread 2026-07-16)
Run every 5 minutes: 9:45-15:15 IST (Mon-Fri)

Opening Range: 9:15-9:45 IST (30m)
Bearish Reject -> Bear Put Spread  (BUY ATM PE + SELL OTM1 PE)
Bullish Reject -> Bull Call Spread (BUY ATM CE + SELL OTM1 CE)
Lower High     -> Bear Put Spread  (confirmation, if no primary)
Higher Low     -> Bull Call Spread (confirmation, if no primary)
Range day skip: price inside OR at 10:15 -> no new entries today
Previous-day filter: if yesterday's net directional move exceeds
PREV_MOVE_THRESHOLD -> no new entries today (added 2026-07-16, see
"Previous-Day Net Move" test in orb_spread.md -- backtest: quiet prior
day PF 1.38 (397 trades) vs big-trend prior day PF 0.98 (395 trades),
essentially a coin-flip loser. Fails OPEN (does not block trading) if
the fetch fails -- this is an added filter on an already-working
strategy, not a risk control, so a fetch outage should not blank the
whole day.

Exit decision is based on NIFTY SPOT movement, not spread premium percent
— matches the backtest methodology exactly (indices-system/strategies/orb_spread.md,
"Debit Spread Structure" section):
    Target: +40 pts spot move in favor | Stop: -25 pts spot move against
    Hard exit: 15:15 IST (changed from 14:30, 2026-07-16 -- see "Hard Exit Time" test)

Spread width: 50 pts (ATM long leg + OTM1 short leg = one real NIFTY
strike interval). Adopted from backtest evidence — a narrow/cheap spread
caps max loss below the strategy's own -25pt stop while barely capping
the +40pt target, producing backtest PF 3.99 (50pt/15%-assumed-cost) vs
naked-long PF 1.14 and a 100pt/35%-cost spread's PF 0.30. Real cost is
whatever the market actually charges at entry, not an assumption — this
script tracks it directly from live leg prices.

Execution: entry uses /api/v1/optionsmultiorder (offset-based ATM/OTM1
resolution against CURRENT spot — correct for opening a new position).
Exit deliberately does NOT reuse optionsmultiorder's offset resolution —
by the time we exit, spot has moved (that's why we're exiting), so
re-resolving "ATM" then could target a different strike than what we
actually hold. Exit closes the exact stored entry symbols via two plain
/api/v1/placeorder calls instead.

State persisted to state/orb_spread_state.json (intraday)
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

# Deployed copies run from strategies/scripts/ (Python Strategy Host), where
# only the script's own directory is on sys.path by default -- add the parent
# strategies/ dir so capital_state.py resolves from both source and deployed
# locations without needing a duplicated copy kept in sync.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capital_state import get_allocated_capital, record_trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("orb_spread")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("OPENALGO_API_KEY", "your_openalgo_api_key_here")
HOST          = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
LOT_SIZE      = 65
OR_MIN        = 30
OR_MAX        = 150
TARGET_PTS    = 40          # NIFTY spot points (matches backtest, not premium %)
STOP_PTS      = 25
SHORT_OFFSET  = "OTM1"      # short leg = one real strike interval from ATM (~50pts)
PREV_MOVE_THRESHOLD = 0.42  # % -- matches the backtest's entry-level median split (2026-07-16)
ENTRY_END     = dtime(12, 0)
HARD_EXIT     = dtime(15, 15)   # changed from 14:30, 2026-07-16 -- see orb_spread.md "Hard Exit Time" test
RANGE_CHK     = dtime(10, 15)
IST           = pytz.timezone("Asia/Kolkata")
STATE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "orb_spread_state.json")

# Debit-spread payoff model for trade-log P&L estimation (mirrors the adopted
# backtest config -- orb_spread.md "Debit Spread Structure", 50pt/15%-cost).
# Real entry mechanics already trade the actual 50pt-interval OTM1 spread via
# broker offset resolution; this is only used to estimate rupee P&L for the
# trade log / ongoing Sharpe-PF-WR tracking (2026-07-18), not the exit decision.
SPREAD_WIDTH  = 50
SPREAD_COST   = SPREAD_WIDTH * 0.15  # 7.5 pts

# Capital-based sizing (2026-07-18, redesigned 2026-07-21 three times, Karan-
# confirmed): bull and bear each get their own FIXED, INDEPENDENT capital
# pool -- (allocated_capital x DEPLOYMENT_CAP_PCT) / 2 each -- rather than
# sharing one combined pool (that was tried first; replaced same day because
# whichever signal fired first could claim most/all of a shared pool,
# starving or zeroing the second same-day entry). Each leg's cap is a
# fraction of allocated_capital, not a frozen rupee figure, so both scale up
# automatically as profit compounds in live mode. See compute_lot_quantity().
DEPLOYMENT_CAP_PCT = 0.80  # max total capital deployed across BOTH positions combined (2 x 40% each), Karan-confirmed 2026-07-21

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
            "bear_position": None, "bull_position": None, "prev_move_pct": None}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), default=str)

# ── OpenAlgo helpers ──────────────────────────────────────────────────────────
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
    return df

def get_prev_day_move_pct():
    """Previous trading day's net directional move as % of that day's open
    (|close - open| / open * 100). Regime filter -- see orb_spread.md
    "Previous-Day Net Move" test for why."""
    today = date.today()
    start = (today - timedelta(days=10)).isoformat()   # buffer for weekends/holidays
    end   = (today - timedelta(days=1)).isoformat()
    r = requests.post(f"{HOST}/api/v1/history",
                      json={"apikey": API_KEY, "symbol": "NIFTY", "exchange": "NSE_INDEX",
                            "interval": "D", "start_date": start, "end_date": end},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success" or not data.get("data"):
        raise RuntimeError(f"History API error: {data}")
    last = data["data"][-1]   # most recent daily bar = yesterday's session
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
    """symbols: list of (symbol, exchange) tuples. Returns {symbol: ltp}."""
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

def estimate_net_debit(option_type):
    """Resolve the ATM/OTM1 leg symbols and fetch their live quotes WITHOUT
    placing an order, so compute_lot_quantity can size against the REAL
    premium about to be paid, not a fixed assumption."""
    expiry = get_current_expiry()
    long_sym = resolve_option_symbol(option_type, "ATM", expiry)
    short_sym = resolve_option_symbol(option_type, SHORT_OFFSET, expiry)
    quotes = get_multiquotes([(long_sym, "NFO"), (short_sym, "NFO")])
    return quotes[long_sym] - quotes[short_sym]

def compute_lot_quantity(net_debit_per_share):
    """Capital-based sizing (2026-07-18, redesigned 2026-07-21 three times,
    Karan-confirmed): bull and bear each get their own FIXED, INDEPENDENT
    capital pool -- (allocated_capital x DEPLOYMENT_CAP_PCT) / 2 -- rather
    than sharing one combined pool. A shared-pool design was tried first the
    same day and replaced: whichever signal fired first could claim most/all
    of the shared amount, starving or zeroing the second same-day entry.
    Fixing each leg's own pool removes that order-of-arrival risk, at the
    cost of some capital efficiency on solo-signal days (a lone entry is now
    capped at half the deployment ceiling, not the full amount -- accepted
    tradeoff). The per-leg cap is a fraction of allocated_capital, not a
    frozen rupee figure, so it scales up automatically as profit compounds
    in live mode, same as everything else in this script.

    Sizes against the REAL live net debit quoted for this entry (see
    estimate_net_debit), not the backtest's modeled SPREAD_COST=7.5pt
    assumption -- that assumption silently sized a 35-lot/2275-share order
    on 2026-07-21 when this function used it directly instead of a real
    quote. risk_per_share is still floored at SPREAD_COST so an
    unrealistically cheap quote can't size UP past what the model expects.

    Can return 0 if even 1 lot doesn't fit the per-leg cap at the real
    quoted premium -- the caller must skip the trade rather than force a
    minimum entry. No separate MAX_LOTS backstop -- removed 2026-07-21
    (Karan-confirmed) since it would silently override the capital-based
    settings; the per-leg capital math above is the sole sizing limit."""
    allocated_capital = get_allocated_capital("orb_spread")
    per_leg_cap = (allocated_capital * DEPLOYMENT_CAP_PCT) / 2
    risk_per_share = max(net_debit_per_share, SPREAD_COST)
    lots = int(per_leg_cap // (risk_per_share * LOT_SIZE))
    return lots * LOT_SIZE

def place_order(symbol, action, quantity):
    r = requests.post(f"{HOST}/api/v1/placeorder",
                      json={"apikey": API_KEY, "strategy": "orb_spread", "symbol": symbol, "exchange": "NFO",
                            "action": action, "quantity": quantity,
                            "pricetype": "MARKET", "product": "MIS"},
                      headers=_headers(), timeout=15)
    r.raise_for_status()
    return r.json()

def open_spread(option_type, quantity):
    """BUY ATM + SELL OTM1 (50pt-wide debit spread), resolved against
    CURRENT spot -- correct for opening a fresh position.
    Returns (long_symbol, short_symbol, long_ltp, short_ltp)."""
    expiry = get_current_expiry()
    r = requests.post(f"{HOST}/api/v1/optionsmultiorder",
                      json={"apikey": API_KEY, "strategy": "orb_spread", "underlying": "NIFTY",
                            "exchange": "NSE_INDEX", "expiry_date": expiry,
                            "legs": [
                                {"offset": "ATM", "option_type": option_type, "action": "BUY", "quantity": quantity},
                                {"offset": SHORT_OFFSET, "option_type": option_type, "action": "SELL", "quantity": quantity},
                            ]},
                      headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(data)
    results = sorted(data["results"], key=lambda x: x["leg"])
    long_sym, short_sym = results[0]["symbol"], results[1]["symbol"]
    quotes = get_multiquotes([(long_sym, "NFO"), (short_sym, "NFO")])
    return long_sym, short_sym, quotes[long_sym], quotes[short_sym]

def close_spread(long_symbol, short_symbol, quantity):
    """Close both legs using their EXACT entry symbols -- NOT re-resolved
    offsets, since spot has moved since entry (that's why we're exiting)
    and re-resolving ATM/OTM1 now could target a different strike. Quantity
    must match what was actually entered (position sizing is no longer a
    fixed constant -- see compute_lot_quantity)."""
    place_order(long_symbol, "SELL", quantity)
    place_order(short_symbol, "BUY", quantity)


def log_closed_trade(direction, reason, pos, spot, pnl_pts, now):
    """Estimate rupee P&L via the adopted spread payoff model and append to
    the trade log (paper AND live) for ongoing Sharpe/PF/win-rate/avg-P&L
    tracking (2026-07-18). Live mode also compounds pnl_rupees into
    allocated_capital -- see capital_state.py."""
    spread_pnl_pts = min(max(pnl_pts, 0), SPREAD_WIDTH) - SPREAD_COST
    qty = pos.get("quantity", LOT_SIZE)
    pnl_rupees = spread_pnl_pts * qty
    record_trade("orb_spread", {
        "date": now.strftime("%Y-%m-%d"),
        "entry_time": pos["entry_time"],
        "direction": direction,
        "signal": pos["signal"],
        "entry_spot": round(pos["entry_spot"], 2),
        "exit_time": now.isoformat(),
        "exit_spot": round(spot, 2),
        "pnl_pts": round(pnl_pts, 2),
        "spread_pnl_pts": round(spread_pnl_pts, 2),
        "pnl_rupees": round(pnl_rupees, 2),
        "reason": reason,
    }, pnl_rupees)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(IST)
    t   = now.time()
    log.info(f"ORB_Spread check — {now.strftime('%H:%M:%S IST')}")

    state = load_state()

    # ── Previous-day net move regime filter (computed once per day) ──────────
    # Retried every tick until it succeeds; never blocks the rest of main() --
    # fails OPEN (entries allowed) if the fetch is unavailable, since this is
    # an added filter on an already-working strategy, not a risk control.
    if state.get("prev_move_pct") is None:
        try:
            state["prev_move_pct"] = get_prev_day_move_pct()
            log.info(f"Prev-day net move = {state['prev_move_pct']:.2f}% "
                     f"(entries allowed only if <= {PREV_MOVE_THRESHOLD}%)")
            save_state(state)
        except Exception as e:
            log.warning(f"Prev-day move fetch failed (will retry next tick): {e}")

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

    orb_high = state["orb_high"]
    orb_low  = state["orb_low"]

    # ── Range day check at 10:15 ──────────────────────────────────────────────
    if t >= RANGE_CHK and not state.get("range_checked"):
        try:
            spot = get_nifty_spot()
            if orb_low < spot < orb_high:
                log.info(f"Range day detected at 10:15 (spot {spot:.2f} inside OR) — no more entries")
                state["range_day"]    = True
                state["range_checked"] = True
                save_state(state)
                return
            state["range_checked"] = True
        except Exception as e:
            log.warning(f"Range day check failed: {e}")

    # ── Exit open positions (spot-points based, matches backtest) ────────────
    for direction, sign in (("bear", 1), ("bull", -1)):
        pos = state.get(f"{direction}_position")
        if not pos:
            continue
        try:
            spot       = get_nifty_spot()
            entry_spot = pos["entry_spot"]
            pnl_pts    = (entry_spot - spot) * sign  # bear profits on spot falling, bull on spot rising

            if pnl_pts >= TARGET_PTS:
                close_spread(pos["long_symbol"], pos["short_symbol"], pos["quantity"])
                log.info(f"EXIT TARGET {direction.upper()} entry_spot={entry_spot:.2f} spot={spot:.2f} (+{pnl_pts:.1f}pts)")
                log_closed_trade(direction, "TARGET", pos, spot, pnl_pts, now)
                state[f"{direction}_position"] = None
            elif pnl_pts <= -STOP_PTS:
                close_spread(pos["long_symbol"], pos["short_symbol"], pos["quantity"])
                log.info(f"EXIT STOP {direction.upper()} entry_spot={entry_spot:.2f} spot={spot:.2f} ({pnl_pts:.1f}pts)")
                log_closed_trade(direction, "STOP", pos, spot, pnl_pts, now)
                state[f"{direction}_position"] = None
            elif t >= HARD_EXIT:
                close_spread(pos["long_symbol"], pos["short_symbol"], pos["quantity"])
                log.info(f"EXIT HARD {direction.upper()} entry_spot={entry_spot:.2f} spot={spot:.2f} ({pnl_pts:.1f}pts)")
                log_closed_trade(direction, "HARD_EXIT", pos, spot, pnl_pts, now)
                state[f"{direction}_position"] = None
            else:
                # Informational net-value check (not the exit decision)
                try:
                    quotes = get_multiquotes([(pos["long_symbol"], "NFO"), (pos["short_symbol"], "NFO")])
                    net_now = quotes[pos["long_symbol"]] - quotes[pos["short_symbol"]]
                    log.info(f"HOLD {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts) "
                             f"net_debit_now={net_now:.2f} (entry {pos['entry_net_debit']:.2f})")
                except Exception:
                    log.info(f"HOLD {direction.upper()} spot={spot:.2f} ({pnl_pts:+.1f}pts)")
        except Exception as e:
            log.warning(f"Exit check failed for {direction}: {e}")

    save_state(state)

    # ── No new entries after 12:00, hard exit, or on a range day ─────────────
    if t > ENTRY_END or t >= HARD_EXIT:
        return
    if state["range_day"]:
        log.info("Range day — no new entries.")
        return
    if state.get("prev_move_pct") is not None and state["prev_move_pct"] > PREV_MOVE_THRESHOLD:
        log.info(f"Prev-day move {state['prev_move_pct']:.2f}% > {PREV_MOVE_THRESHOLD}% threshold — no new entries.")
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
        i  = len(sig) - 1

        bear_signal = bull_signal = None

        if c[i-1] > orb_high and c[i] < orb_high and c[i] < o[i]:
            bear_signal = "BearishReject"
        if c[i-1] < orb_low and c[i] > orb_low and c[i] > o[i]:
            bull_signal = "BullishReject"
        if bear_signal is None and i >= 2:
            if h[i] < h[i-1] < h[i-2] and c[i] < orb_high:
                bear_signal = "LowerHigh"
        if bull_signal is None and i >= 2:
            if lo[i] > lo[i-1] > lo[i-2] and c[i] > orb_low:
                bull_signal = "HigherLow"

        spot = float(c[i])

        # ── Enter Bear Put Spread ─────────────────────────────────────────────
        if bear_signal and not state["bear_traded"] and not state["bear_position"]:
            try:
                net_debit_est = estimate_net_debit("PE")
                quantity = compute_lot_quantity(net_debit_est)
                if quantity <= 0:
                    log.info(f"BEAR signal {bear_signal} skipped — even 1 lot doesn't fit "
                             f"the per-leg capital pool at est net_debit {net_debit_est:.2f}. "
                             f"Not marking bear_traded -- will retry next tick.")
                else:
                    long_sym, short_sym, long_ltp, short_ltp = open_spread("PE", quantity)
                    net_debit = long_ltp - short_ltp
                    log.info(f"ENTRY BEAR PUT SPREAD {bear_signal} | +{long_sym} -{short_sym} | "
                             f"spot={spot:.2f} | net_debit={net_debit:.2f} (sized against est {net_debit_est:.2f}) | qty={quantity}")
                    if net_debit > net_debit_est * 1.5:
                        log.warning(f"Real fill net_debit {net_debit:.2f} is well above the {net_debit_est:.2f}pt "
                                    f"estimate used for sizing -- price moved between the sizing quote and the "
                                    f"fill; actual capital committed this trade (~Rs {net_debit * quantity:,.0f}) "
                                    f"may exceed the intended risk budget.")
                    state["bear_traded"]   = True
                    state["bear_position"] = {
                        "long_symbol": long_sym, "short_symbol": short_sym,
                        "entry_spot": spot, "entry_net_debit": net_debit,
                        "entry_time": now.isoformat(), "signal": bear_signal,
                        "quantity": quantity,
                    }
            except Exception as e:
                log.error(f"Bear entry failed: {e}")

        # ── Enter Bull Call Spread ─────────────────────────────────────────────
        if bull_signal and not state["bull_traded"] and not state["bull_position"]:
            try:
                net_debit_est = estimate_net_debit("CE")
                quantity = compute_lot_quantity(net_debit_est)
                if quantity <= 0:
                    log.info(f"BULL signal {bull_signal} skipped — even 1 lot doesn't fit "
                             f"the per-leg capital pool at est net_debit {net_debit_est:.2f}. "
                             f"Not marking bull_traded -- will retry next tick.")
                else:
                    long_sym, short_sym, long_ltp, short_ltp = open_spread("CE", quantity)
                    net_debit = long_ltp - short_ltp
                    log.info(f"ENTRY BULL CALL SPREAD {bull_signal} | +{long_sym} -{short_sym} | "
                             f"spot={spot:.2f} | net_debit={net_debit:.2f} (sized against est {net_debit_est:.2f}) | qty={quantity}")
                    if net_debit > net_debit_est * 1.5:
                        log.warning(f"Real fill net_debit {net_debit:.2f} is well above the {net_debit_est:.2f}pt "
                                    f"estimate used for sizing -- price moved between the sizing quote and the "
                                    f"fill; actual capital committed this trade (~Rs {net_debit * quantity:,.0f}) "
                                    f"may exceed the intended risk budget.")
                    state["bull_traded"]   = True
                    state["bull_position"] = {
                        "long_symbol": long_sym, "short_symbol": short_sym,
                        "entry_spot": spot, "entry_net_debit": net_debit,
                        "entry_time": now.isoformat(), "signal": bull_signal,
                        "quantity": quantity,
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
        time.sleep(300)  # 5 minutes
