"""
Strategy Metrics Service — Sharpe ratio, profit factor, win rate, average
P&L per strategy, plus rolling correlation between strategies.

Implements Karan's ongoing-tracking requirement (2026-07-18): Sharpe/PF/
win-rate/avg-P&L must be tracked continuously through paper trading and into
live trading, not computed on demand -- see agents/friday/memory/decisions.md.

Reads the structured trade-log CSVs each live/paper script writes via
capital_state.record_trade() (strategies/capital_state.py) and recomputes
from the full history each run -- trade counts stay small enough that this
is simpler/safer than incremental aggregation. Writes a single markdown
report to the vault, overwritten each run (not one file per day).
"""

import itertools
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.normpath(os.path.join(_SERVICES_DIR, "..", "strategies"))
LOGS_DIR = os.path.join(STRATEGIES_DIR, "logs")
CAPITAL_STATE_FILE = os.path.join(STRATEGIES_DIR, "scripts", "state", "capital_allocation.json")

VAULT_ROOT = os.path.normpath(os.path.join(STRATEGIES_DIR, "..", ".."))
PORTFOLIO_TRACKING_PATH = os.path.join(VAULT_ROOT, "portfolio-tracking.md")

TRADING_DAYS_PER_YEAR = 252
MIN_TRADES_FOR_GATE = 20   # matches this vault's existing evidence-gate convention (scorecard.md)
CORRELATION_WINDOW = 30    # trading days, rolling -- named default (2026-07-18), easy to change

STRATEGIES = {
    "orb_spread": {
        "name": "ORB_Spread",
        "spec": "indices-system/strategies/orb_spread.md",
    },
    "ema_regime_crossover": {
        "name": "EMA Regime Crossover",
        "spec": "equities-system/strategies/ema_regime_crossover.md",
    },
    "ema_regime_crossover_swing": {
        "name": "EMA Regime Crossover — Swing",
        "spec": "equities-system/strategies/ema_regime_crossover_swing.md",
    },
}


def load_capital_state():
    with open(CAPITAL_STATE_FILE) as f:
        return json.load(f)


def load_trades(strategy_key):
    path = os.path.join(LOGS_DIR, f"{strategy_key}_trades.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["exit_date"] = df["exit_time"].dt.date
    return df


def annualised_sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2 or daily_returns.std(ddof=1) == 0:
        return float("nan")
    return daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_strategy_metrics(strategy_key, mode_filter=None):
    """mode_filter: None (all trades), 'paper', or 'live' -- lets the report
    break out paper vs live evidence separately once a strategy goes live."""
    df = load_trades(strategy_key)
    if mode_filter and not df.empty:
        df = df[df["mode"] == mode_filter]

    capital_state = load_capital_state()
    allocated_capital = capital_state[strategy_key]["allocated_capital"]

    if df.empty:
        return {
            "n_trades": 0, "win_rate": None, "profit_factor": None, "avg_pnl": None,
            "sharpe": None, "daily_returns": pd.Series(dtype=float),
            "allocated_capital": allocated_capital,
        }

    n = len(df)
    wins = int((df["pnl_rupees"] > 0).sum())
    win_rate = wins / n * 100
    gross_win = df.loc[df["pnl_rupees"] > 0, "pnl_rupees"].sum()
    gross_loss = abs(df.loc[df["pnl_rupees"] <= 0, "pnl_rupees"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_pnl = df["pnl_rupees"].mean()

    daily_pnl = df.groupby("exit_date")["pnl_rupees"].sum()
    daily_returns = daily_pnl / allocated_capital
    sharpe = annualised_sharpe(daily_returns)

    return {
        "n_trades": n, "win_rate": win_rate, "profit_factor": profit_factor,
        "avg_pnl": avg_pnl, "sharpe": sharpe, "daily_returns": daily_returns,
        "allocated_capital": allocated_capital,
    }


def compute_correlation(daily_returns_a: pd.Series, daily_returns_b: pd.Series, window=CORRELATION_WINDOW):
    """Outer-joined on date, 0-filled for days only one strategy traded --
    both are intraday-flat strategies, so a day with no trade is a genuine
    zero return, not a missing observation."""
    combined = pd.DataFrame({"a": daily_returns_a, "b": daily_returns_b}).fillna(0.0)
    if len(combined) < 2:
        return {"full_history": None, "rolling": None, "n_days": len(combined)}
    full = combined["a"].corr(combined["b"])
    rolling = None
    if len(combined) >= window:
        rolling = combined["a"].rolling(window).corr(combined["b"]).iloc[-1]
    return {"full_history": full, "rolling": rolling, "n_days": len(combined)}


def _fmt_pf(pf):
    if pf is None:
        return "—"
    return "∞ (no losses yet)" if pf == float("inf") else f"{pf:.2f}"


def _fmt_sharpe(sharpe):
    if sharpe is None or (isinstance(sharpe, float) and np.isnan(sharpe)):
        return "Accumulating (need >=2 trading days)"
    return f"{sharpe:.2f}"


def _gate_note(n_trades):
    return "" if n_trades >= MIN_TRADES_FOR_GATE else f" *(Accumulating, <{MIN_TRADES_FOR_GATE})*"


def build_report() -> str:
    metrics = {key: compute_strategy_metrics(key) for key in STRATEGIES}
    # All-pairs correlation -- generalized from an original hardcoded 2-strategy
    # comparison once a 3rd strategy (ema_regime_crossover_swing) existed.
    correlations = {
        (a, b): compute_correlation(metrics[a]["daily_returns"], metrics[b]["daily_returns"])
        for a, b in itertools.combinations(STRATEGIES, 2)
    }

    lines = [
        "# Portfolio Tracking — Ongoing Sharpe, PF, Win Rate, Avg P&L, Correlation",
        "",
        f"*Auto-generated by `OpenAlgo/services/strategy_metrics_service.py` — last run "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Recomputed from the full trade-log "
        f"CSV history each run, not incrementally. Implements the tracking goal set 2026-07-18 "
        f"— see `agents/friday/memory/decisions.md`.*",
        "",
        "**Do not hand-edit this file** — it is overwritten on every scheduled run. Narrative "
        "context and evidence discussion belongs in each strategy's own spec or in "
        "`agents/friday/memory/decisions.md`.",
        "",
        "## Per-Strategy Metrics",
        "",
        "| Strategy | Trades | Win Rate | Profit Factor | Avg P&L | Sharpe (goal ≥ 2.0) | Allocated Capital |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, info in STRATEGIES.items():
        m = metrics[key]
        if m["n_trades"] == 0:
            lines.append(
                f"| {info['name']} | 0 | — | — | — | — | Rs {m['allocated_capital']:,.0f} |"
            )
            continue
        lines.append(
            f"| {info['name']} | {m['n_trades']}{_gate_note(m['n_trades'])} | "
            f"{m['win_rate']:.1f}% | {_fmt_pf(m['profit_factor'])} | "
            f"Rs {m['avg_pnl']:+,.0f} | {_fmt_sharpe(m['sharpe'])} | "
            f"Rs {m['allocated_capital']:,.0f} |"
        )

    lines += [
        "",
        "## Cross-Strategy Correlation (all pairs)",
        "",
    ]
    for (a, b), corr in correlations.items():
        pair_name = f"{STRATEGIES[a]['name']} vs {STRATEGIES[b]['name']}"
        lines.append(f"**{pair_name}:**")
        if corr["full_history"] is None:
            lines.append("- Not enough overlapping trading days yet to compute correlation.")
        else:
            rolling_str = (
                f"{corr['rolling']:.2f}" if corr["rolling"] is not None
                else f"Accumulating (need {CORRELATION_WINDOW} overlapping trading days, have {corr['n_days']})"
            )
            lines.append(f"- Full-history Pearson correlation (daily returns, {corr['n_days']} overlapping days): **{corr['full_history']:.2f}**")
            lines.append(f"- Rolling {CORRELATION_WINDOW}-trading-day correlation: **{rolling_str}**")
        lines.append("")

    lines += [
        "",
        "## Caveats",
        "",
        "- Sharpe divisor is each strategy's *current* allocated capital (compounds with live P&L "
        "once `mode == \"live\"`) — an approximation while capital is actively changing, not an "
        "exact figure.",
        f"- Correlation basis: daily P&L ÷ that day's allocated capital, outer-joined on date with "
        f"0-fill for days only one strategy traded. {CORRELATION_WINDOW}-day rolling window is a "
        f"named default (2026-07-18) — flag if you want it changed.",
        f"- Trade counts below {MIN_TRADES_FOR_GATE} are flagged \"Accumulating\", matching this "
        f"vault's existing evidence-gate convention (`indices-system/scorecard.md`) — not "
        f"suppressed, just not yet meaningful.",
        "- This is a paper-and-live-combined view by default (no mode filter). All trades right "
        "now are paper — see each strategy's own spec for forward-test trade counts and context.",
        "",
        "---",
        "",
        "*See also: " + " · ".join(f"[[{info['spec']}]]" for info in STRATEGIES.values())
        + " · [[agents/friday/memory/decisions.md]]*",
    ]
    return "\n".join(lines) + "\n"


def run_metrics_report():
    """Entry point for the scheduled job (and for on-demand runs/testing)."""
    report = build_report()
    with open(PORTFOLIO_TRACKING_PATH, "w") as f:
        f.write(report)
    return PORTFOLIO_TRACKING_PATH


if __name__ == "__main__":
    path = run_metrics_report()
    print(f"Wrote {path}")
