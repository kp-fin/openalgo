"""
Shared capital-allocation and trade-log helpers for ORB_Spread and EMA Regime
Crossover live/paper scripts.

Implements Karan's capital plan (2026-07-18): Rs 50,000 allocated to ORB_Spread,
Rs 15,000 to EMA Regime Crossover, both live-trading-only (paper P&L never
touches allocated capital). Live profit/loss compounds into allocated_capital
per closed trade, bidirectionally, no floor. See
agents/friday/memory/decisions.md and both strategies' specs for the full
rationale.

capital_allocation.json's "mode" field is a manual switch -- flip
"paper" -> "live" per strategy when it actually starts live trading. No
auto-detection from OpenAlgo's analyzer/sandbox mode, by design.
"""

import csv
import json
import os
from datetime import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
CAPITAL_STATE_FILE = os.path.join(_BASE, "scripts", "state", "capital_allocation.json")
LOGS_DIR = os.path.join(_BASE, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

TRADE_LOG_COLUMNS = {
    "orb_spread": [
        "date", "entry_time", "direction", "signal", "entry_spot", "exit_time",
        "exit_spot", "pnl_pts", "exit_net_debit", "pnl_rupees", "reason", "mode",
    ],
    "ema_regime_crossover_swing": [
        "date", "symbol", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "qty", "pnl_rupees", "reason", "mode",
    ],
    "ema_regime_crossover_swing_cnc_top15nifty100": [
        "date", "symbol", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "qty", "pnl_rupees", "charges_rupees",
        "net_pnl_rupees", "reason", "hold_days", "group", "mode",
    ],
    "hh_hl_pullback_breakout_nifty200": [
        "date", "symbol", "direction", "entry_price",
        "exit_time", "exit_price", "qty", "pnl_rupees", "charges_rupees",
        "net_pnl_rupees", "reason", "hold_days", "group", "mode",
    ],
    "gap_and_go": [
        "date", "symbol", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "qty", "pnl_rupees", "reason", "mode",
    ],
}


def load_capital_state():
    with open(CAPITAL_STATE_FILE) as f:
        return json.load(f)


def save_capital_state(state):
    with open(CAPITAL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_allocated_capital(strategy_key):
    """Current allocated capital for a strategy. Call fresh on every sizing
    decision, not cached -- it changes per live trade close."""
    return load_capital_state()[strategy_key]["allocated_capital"]


def record_trade(strategy_key, row: dict, pnl_rupees: float):
    """Append a closed trade to the strategy's trade-log CSV (paper AND live --
    this is the ongoing Sharpe/PF/win-rate/avg-P&L evidence trail). Only when
    mode == "live" does pnl_rupees compound into allocated_capital -- paper
    trades are recorded for tracking but never move capital, per Karan's
    explicit "whatever profits made during LIVE trading" framing (2026-07-18).
    """
    state = load_capital_state()
    mode = state[strategy_key]["mode"]

    row = dict(row)
    row["mode"] = mode

    log_path = os.path.join(LOGS_DIR, f"{strategy_key}_trades.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS[strategy_key])
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    if mode == "live":
        state[strategy_key]["allocated_capital"] += pnl_rupees
        state[strategy_key]["last_updated"] = datetime.now().isoformat()
        save_capital_state(state)
